import base64
import hashlib
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from fastapi import APIRouter, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session
from src.api.models import LabelingContext, Request, BaseContext, Labeler
from src.config import RECORDINGS_PATH, Sam2Checkpoints, Template, templates, FRAMES_PATH
from src.db import CalibrationRecording, engine
from src.db.models import Recording, Annotation
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
            request.app.labeler = Labeler(calibration_recording=cal_rec)
            
            labeling_context = LabelingContext(
                request=request,
                sim_room=cal_rec.sim_room,
                frame_count=request.app.labeler.frame_count,
            )

            if is_hx_request(request):
                return templates.TemplateResponse(Template.LABELER, labeling_context.to_dict())
            return templates.TemplateResponse(Template.INDEX, labeling_context.to_dict())
        except Exception as e:
            return Response(status_code=500, content=f"Failed to start labeling: {e!s}")


@router.get("/", response_class=HTMLResponse)
async def labeling(request: Request, calibration_id: int):
    if not request.app.labeler:
        return RedirectResponse(status_code=307, url="/simrooms")

    if request.app.labeler.calibration_recording.id != calibration_id:
        return RedirectResponse(status_code=307, url="/simrooms")

    with Session(engine) as session:
        cal_rec = session.query(CalibrationRecording).filter(CalibrationRecording.id == calibration_id).first()

        if not cal_rec:
            return Response(status_code=404, content="Error: Calibration recording not found")

        labeling_context = LabelingContext(
            request=request,
            sim_room=cal_rec.sim_room,
            frame_count=request.app.labeler.frame_count,
        )

        if is_hx_request(request):
            return templates.TemplateResponse(Template.LABELER, labeling_context.to_dict())
        return templates.TemplateResponse(Template.INDEX, labeling_context.to_dict())

@router.get("/seek", response_class=Response)
async def seek(request: Request, percent: float):
    if not request.app.labeler:
        return RedirectResponse(status_code=307, url="/simrooms")

    frame = request.app.labeler.seek(percent)
    ret, encoded_png = cv2.imencode(".png", frame)
    if not ret:
        return Response(status_code=500, content="Error: Failed to encode frame")
    
    return Response(content=encoded_png.tobytes(), media_type="image/png")

@dataclass
class LabelingAnnotationsContext(BaseContext):
    annotations: list[Annotation] = field(default_factory=list)

@router.get("/annotations", response_class=Response)
async def annotations(request: Request, class_id: int | None = None):
    if not request.app.labeler:
        return RedirectResponse(status_code=307, url="/simrooms")
    
    
    context = LabelingAnnotationsContext(request=request)

    if class_id is not None:
        with Session(engine) as session:
            cal_rec_id = request.app.labeler.calibration_recording.id
            context.annotations = session.query(Annotation).filter(
                Annotation.calibration_recording_id == cal_rec_id,
                Annotation.sim_room_class_id == class_id,
            ).all()

    return templates.TemplateResponse(Template.ANNOTATIONS, context.to_dict())

@router.delete("/annotations/{annotation_id}", response_class=Response)
async def delete_calibration_annotation(request: Request, annotation_id: int):
    with Session(engine) as session:
        annotation = (
            session.query(Annotation)
            .filter(Annotation.id == annotation_id)
            .first()
        )
        
        if not annotation:
            return Response(status_code=404, content="Annotation not found")
        
        sim_room_class_id = annotation.sim_room_class_id
        session.delete(annotation)
        session.commit()
        
        return await annotations(request, sim_room_class_id)

