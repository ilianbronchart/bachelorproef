import base64
from dataclasses import dataclass

import cv2
from fastapi import APIRouter, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session
from src.api.models import LabelingContext, Request
from src.config import RECORDINGS_PATH, Sam2Checkpoints, Template, templates
from src.db import CalibrationRecording, engine
from src.db.models import Recording
from src.logic.inference.sam_2 import load_sam2_predictor, predict_sam2
from src.utils import cv2_get_frame, cv2_video_frame_count, cv2_video_resolution, is_hx_request

router = APIRouter(prefix="/labeling")


@router.post("/", response_class=JSONResponse)
async def start_labeling(request: Request, calibration_id: int):
    with Session(engine) as session:
        cal_rec = session.query(CalibrationRecording).filter(CalibrationRecording.id == calibration_id).first()

        if not cal_rec:
            return Response(status_code=404, content="Error: Calibration recording not found")

        try:
            video_path = RECORDINGS_PATH / (cal_rec.recording_uuid + ".mp4")
            request.app.labeling_context = LabelingContext(
                request=request,
                calibration_recording=cal_rec,
                sim_room=cal_rec.sim_room,
                recording=cal_rec.recording,
                predictor=load_sam2_predictor(Sam2Checkpoints.LARGE),
                classes=cal_rec.sim_room.classes,
                annotations=cal_rec.annotations,
                frame_count=cv2_video_frame_count(video_path),
                resolution=cv2_video_resolution(video_path, flip=True),
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
        video_path = RECORDINGS_PATH / (recording.uuid + ".mp4")
        frame = cv2_get_frame(video_path, frame_idx)
        request.app.labeling_context.predictor.set_image(frame)

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
        mask = mask.repeat(3, axis=0).transpose(1, 2, 0) * 255
        _, encoded_img = cv2.imencode(".png", mask)

        if mask is None:
            return Response(status_code=400, content="Error: Segmentation had no results")
        
        return JSONResponse({"mask": base64.b64encode(encoded_img.tobytes()).decode("utf-8"), "bounding_box": bounding_box})
    except Exception as e:
        return Response(status_code=500, content=f"Failed to segment frame: {e!s}")
