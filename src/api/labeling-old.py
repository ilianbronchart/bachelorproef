import base64
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from fastapi import APIRouter, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session
from src.api.models import LabelingContext, Request
from src.config import RECORDINGS_PATH, Sam2Checkpoints, Template, templates, FRAMES_PATH
from src.db import CalibrationRecording, engine
from src.db.models import Recording
from src.logic.inference.sam_2 import load_sam2_predictor, load_sam2_video_predictor, predict_sam2
from src.utils import cv2_video_frame_count, cv2_video_resolution, is_hx_request, get_frame_from_dir, extract_frames_to_dir

router = APIRouter(prefix="/labeling")


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@router.post("/", response_class=JSONResponse)
async def start_labeling(request: Request, calibration_id: int):
    with Session(engine) as session:
        cal_rec = session.query(CalibrationRecording).filter(CalibrationRecording.id == calibration_id).first()

        if not cal_rec:
            return Response(status_code=404, content="Error: Calibration recording not found")

        try:
            video_path = RECORDINGS_PATH / (cal_rec.recording_uuid + ".mp4")

            # extract_frames_to_dir(video_path, FRAMES_PATH)

            request.app.labeling_context = LabelingContext(
                request=request,
                calibration_recording=cal_rec,
                selected_sim_room=cal_rec.sim_room,
                recording=cal_rec.recording,
                predictor=load_sam2_predictor(Sam2Checkpoints.LARGE),
                classes=cal_rec.sim_room.classes,
                annotations=cal_rec.annotations,
                frame_count=cv2_video_frame_count(video_path),
                resolution=cv2_video_resolution(video_path, flip=True),
                current_frame=get_frame_from_dir(0, FRAMES_PATH),
            )
        except Exception as e:
            return Response(status_code=500, content=f"Failed to start labeling: {e!s}")

        if is_hx_request(request):
            return templates.TemplateResponse(Template.LABELER, request.app.labeling_context.to_dict())
        return templates.TemplateResponse(Template.INDEX, request.app.labeling_context.to_dict())


@router.get("/", response_class=HTMLResponse)
async def labeling(request: Request, calibration_id: int):
    if not request.app.labeling_context:
        return RedirectResponse(status_code=307, url="/simrooms")

    if request.app.labeling_context.calibration_recording.id != calibration_id:
        return RedirectResponse(status_code=307, url="/simrooms")

    with Session(engine) as session:
        cal_rec = session.query(CalibrationRecording).filter(CalibrationRecording.id == calibration_id).first()

        if not cal_rec:
            return Response(status_code=404, content="Error: Calibration recording not found")

        request.app.labeling_context.classes = cal_rec.sim_room.classes
        request.app.labeling_context.annotations = cal_rec.annotations
        print(len(request.app.labeling_context.annotations))

        if is_hx_request(request):
            return templates.TemplateResponse(Template.LABELER, request.app.labeling_context.to_dict())
        return templates.TemplateResponse(Template.INDEX, request.app.labeling_context.to_dict())


@dataclass
class SegmentationRequestBody:
    calibration_id: int

    points: list[tuple[int, int]]
    """Segmentation Points: List of points (x, y)."""

    labels: list[int]
    """Segmentation Point Labels: 1 for positive, 0 for negative."""

    frame_idx: int


@router.get("/frames/{frame_idx}", response_class=Response)
async def get_frame(request: Request, frame_idx: int):
    """Retrieve a frame from a recording"""
    if not request.app.labeling_context:
        return RedirectResponse(status_code=307, url="/simrooms")

    recording: Recording = request.app.labeling_context.recording
    if recording is None:
        return Response(status_code=404, content="Error: Recording not found")

    try:
        frame = get_frame_from_dir(frame_idx, FRAMES_PATH)
        request.app.labeling_context.predictor.set_image(frame)
        request.app.labeling_context.current_frame = frame

        ret, encoded_png = cv2.imencode(".png", frame)
        if not ret:
            return Response(status_code=500, content="Error: Failed to encode frame")

        return Response(content=encoded_png.tobytes(), media_type="image/png")
    except IndexError:
        return Response(status_code=404, content="Error: Frame not found")
    except ValueError:
        return Response(status_code=400, content="Error: Invalid frame index")
    except Exception:
        return Response(status_code=500, content="Error: Something went wrong, please try again later")


@router.post("/segmentation", response_class=JSONResponse)
async def segmentation(request: Request, body: SegmentationRequestBody):
    if not request.app.labeling_context:
        return RedirectResponse(status_code=307, url="/simrooms")

    if request.app.labeling_context.calibration_recording.id != body.calibration_id:
        return RedirectResponse(status_code=307, url="/simrooms")

    with Session(engine) as session:
        cal_rec = session.query(CalibrationRecording).filter(CalibrationRecording.id == body.calibration_id).first()

        if not cal_rec:
            return Response(status_code=404, content="Error: Calibration recording not found")

    try:
        mask, bounding_box = predict_sam2(
            predictor=request.app.labeling_context.predictor,
            points=body.points,
            points_labels=body.labels,
        )
        if mask is None or bounding_box is None:
            return Response(status_code=400, content="Error: Segmentation had no results")

        mask = mask.repeat(3, axis=0).transpose(1, 2, 0) * 255
        _, encoded_mask = cv2.imencode(".png", mask)
        mask_base64 = base64.b64encode(encoded_mask.tobytes()).decode("utf-8")

        x1, y1, x2, y2 = bounding_box
        frame_crop = request.app.labeling_context.current_frame[y1:y2, x1:x2]
        _, encoded_frame_crop = cv2.imencode(".png", frame_crop)
        frame_crop_base64 = base64.b64encode(encoded_frame_crop.tobytes()).decode("utf-8")

        return JSONResponse({"mask": mask_base64, "bounding_box": bounding_box, "frame_crop": frame_crop_base64})
    except Exception as e:
        return Response(status_code=500, content=f"Failed to segment frame: {e!s}")

def save_segmentation(frame_idx: int, seg_data: dict, output_dir: str):
    try:
        # Define the output file path. Adjust the path as needed.
        file_path = os.path.join(output_dir, f"{frame_idx:05d}.npz")
        np.savez_compressed(file_path, **seg_data)
        # Optionally, confirm the file was saved
    except Exception as e:
        logger.exception(f"Error saving segmentation for frame {frame_idx}: {e}")
    finally:
        # Free up memory.
        del seg_data

@router.post("/tracking", response_class=JSONResponse)
async def tracking(request: Request, calibration_id: int):
    # if not request.app.labeling_context:
    #     return RedirectResponse(status_code=307, url="/simrooms")

    # if request.app.labeling_context.calibration_recording.id != calibration_id:
    #     return RedirectResponse(status_code=307, url="/simrooms")

    with Session(engine) as session:
        cal_rec = session.query(CalibrationRecording).filter(CalibrationRecording.id == calibration_id).first()

        if not cal_rec:
            return Response(status_code=404, content="Error: Calibration recording not found")

        annotations = cal_rec.annotations

        video_predictor = load_sam2_video_predictor(Sam2Checkpoints.LARGE)
        
        # del request.app.labeling_context.predictor

        inference_state = video_predictor.init_state(video_path=str(FRAMES_PATH), async_loading_frames=True)

        for annotation in annotations:
            point_labels = annotation.point_labels
            points = [(int(point_label.x), int(point_label.y)) for point_label in point_labels]
            labels = [point_label.label for point_label in point_labels]

            video_predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=annotation.frame_idx,
                obj_id=annotation.sim_room_class.id,
                points=points,
                labels=labels,
            )

        # Define output directory for segmentation results.
        output_dir = "./segmentation_results"
        # empty the directory
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, file))
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Segmentation output directory set to {output_dir}")

        # Use ThreadPoolExecutor to offload saving to disk.
        futures = []
        with ProcessPoolExecutor() as executor:
            with torch.amp.autocast('cuda'):  # type: ignore
                for out_frame_idx, class_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
                    seg_data = {}
                    for i, out_obj_id in enumerate(class_ids):
                        mask = (out_mask_logits[i] > 0.5).detach()
                        if mask.any():
                            seg_data[str(out_obj_id)] = mask.cpu().numpy()
                    
                    future = executor.submit(save_segmentation, out_frame_idx, seg_data, output_dir)
                    futures.append(future)

    logger.info("Tracking complete")
    return JSONResponse(content={"status": "tracking complete"})