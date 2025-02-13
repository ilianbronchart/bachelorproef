import base64
from dataclasses import dataclass

import cv2
from fastapi import APIRouter, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session
from src.api.models import LabelingContext, Request
from src.config import FAST_SAM_CHECKPOINT, RECORDINGS_PATH, Template, templates
from src.db import CalibrationRecording, engine
from src.logic.inference.fastsam import segment
from src.utils import cv2_get_frame, cv2_video_frame_count, cv2_video_resolution, is_hx_request
from ultralytics import FastSAM

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
                model=FastSAM(str(FAST_SAM_CHECKPOINT)),
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

    video_path = RECORDINGS_PATH / (cal_rec.recording_uuid + ".mp4")
    frame = cv2_get_frame(video_path, body.frame_idx)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mask, bounding_box = segment(
        image=frame,  # type: ignore
        model=request.app.labeling_context.model,
        points=body.points,
        point_labels=body.labels,
    )

    if mask is None:
        print("Error: Segmentation had no results")
        return Response(status_code=400, content="Error: Segmentation had no results")

    mask = mask.repeat(3, axis=0).transpose(1, 2, 0) * 255
    _, encoded_img = cv2.imencode(".png", mask)

    # save the mask locally for debugging
    with open("mask.png", "wb") as f:
        f.write(encoded_img.tobytes())

    return JSONResponse({"mask": base64.b64encode(encoded_img.tobytes()).decode("utf-8"), "bounding_box": bounding_box})
