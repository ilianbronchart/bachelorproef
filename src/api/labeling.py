from dataclasses import dataclass, field

import cv2
from fastapi import APIRouter, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session
from src.api.models import BaseContext, Labeler, LabelingContext, Request, LabelingAnnotationsContext
from src.config import Template, templates
from src.db import CalibrationRecording, engine
from src.db.models import Annotation
from src.utils import (
    is_hx_request,
)

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

@router.get("/point_labels", response_class=JSONResponse)
async def point_labels(request: Request, frame_idx: int):
    if not request.app.labeler:
        return RedirectResponse(status_code=307, url="/simrooms")

    with Session(engine) as session:
        cal_rec_id = request.app.labeler.calibration_recording.id
        annotations = session.query(Annotation).filter(
            Annotation.calibration_recording_id == cal_rec_id,
            Annotation.frame_idx == frame_idx
        ).all()

        point_labels = []
        for annotation in annotations:
            for point_label in annotation.point_labels:
                point_label_dict = point_label.to_dict()
                point_label_dict["class_id"] = annotation.sim_room_class_id # type: ignore
                point_labels.append(point_label_dict)

        return JSONResponse(content={"point_labels": point_labels})


@router.get("/seek", response_class=Response)
async def seek(request: Request, frame_idx: int):
    if not request.app.labeler:
        return RedirectResponse(status_code=307, url="/simrooms")

    with Session(engine) as session:
        request.app.labeler.seek(frame_idx)
        calibration_recording_id = request.app.labeler.calibration_recording.id

        annotations = session.query(Annotation).filter(
            Annotation.calibration_recording_id == calibration_recording_id,
            Annotation.frame_idx == frame_idx
        ).all()

        frame = request.app.labeler.get_overlay(annotations)
        ret, encoded_png = cv2.imencode(".png", frame)
        if not ret:
            return Response(status_code=500, content="Error: Failed to encode frame")
        
        return Response(content=encoded_png.tobytes(), media_type="image/png")


# @router.post("/segmentation", response_class=JSONResponse)
# async def segmentation(request: Request, body: SegmentationRequestBody):
#     if not request.app.labeler:
#         return RedirectResponse(status_code=307, url="/simrooms")

#     if request.app.labeling_context.calibration_recording.id != body.calibration_id:
#         return RedirectResponse(status_code=307, url="/simrooms")

#     with Session(engine) as session:
#         cal_rec = session.query(CalibrationRecording).filter(CalibrationRecording.id == body.calibration_id).first()

#         if not cal_rec:
#             return Response(status_code=404, content="Error: Calibration recording not found")

#     try:
#         mask, bounding_box = predict_sam2(
#             predictor=request.app.labeling_context.predictor,
#             points=body.points,
#             points_labels=body.labels,
#         )
#         if mask is None or bounding_box is None:
#             return Response(status_code=400, content="Error: Segmentation had no results")

#         mask = mask.repeat(3, axis=0).transpose(1, 2, 0) * 255
#         _, encoded_mask = cv2.imencode(".png", mask)
#         mask_base64 = base64.b64encode(encoded_mask.tobytes()).decode("utf-8")

#         x1, y1, x2, y2 = bounding_box
#         frame_crop = request.app.labeling_context.current_frame[y1:y2, x1:x2]
#         _, encoded_frame_crop = cv2.imencode(".png", frame_crop)
#         frame_crop_base64 = base64.b64encode(encoded_frame_crop.tobytes()).decode("utf-8")

#         return JSONResponse({"mask": mask_base64, "bounding_box": bounding_box, "frame_crop": frame_crop_base64})
#     except Exception as e:
#         return Response(status_code=500, content=f"Failed to segment frame: {e!s}")

@router.get("/annotations", response_class=Response)
async def annotations(request: Request, class_id: int | None = None):
    if not request.app.labeler:
        return RedirectResponse(status_code=307, url="/simrooms")

    context = LabelingAnnotationsContext(request=request)

    if class_id is not None:
        with Session(engine) as session:
            cal_rec_id = request.app.labeler.calibration_recording.id
            context.annotations = (
                session.query(Annotation)
                .filter(
                    Annotation.calibration_recording_id == cal_rec_id,
                    Annotation.sim_room_class_id == class_id,
                )
                .all()
            )

    return templates.TemplateResponse(Template.ANNOTATIONS, context.to_dict())


@router.delete("/annotations/{annotation_id}", response_class=Response)
async def delete_calibration_annotation(request: Request, annotation_id: int):
    with Session(engine) as session:
        annotation = session.query(Annotation).filter(Annotation.id == annotation_id).first()

        if not annotation:
            return Response(status_code=404, content="Annotation not found")

        sim_room_class_id = annotation.sim_room_class_id
        session.delete(annotation)
        session.commit()

        return await annotations(request, sim_room_class_id)
