from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, List

import src.logic.glasses as glasses
from fastapi import APIRouter, Form, Response
from fastapi.responses import HTMLResponse, JSONResponse
from src.config import BaseContext, Request, Template, templates, FAST_SAM_CHECKPOINT
from src.core.utils import is_hx_request, base64_to_numpy
from src.db import Recording, CalibrationRecording, SimRoomClass, Annotation, PointLabel, SimRoom
from sqlalchemy.orm import Session, joinedload
from src.db.db import engine
from src.logic.inference import fastsam
from ultralytics import FastSAM

router = APIRouter(prefix="/labeling")


@dataclass
class LabelingContext(BaseContext):
    recording_uuid: str | None = None
    calibration_recording: CalibrationRecording | None = None
    classes: List[dict[str, Any]] = None  # [{label: str, color: str}]
    annotations: List[dict[str, Any]] = None  # [{label: str, points: list[tuple[int, int]], point_labels: list[int]}]
    content: str = Template.LABELER


@dataclass
class PointAnnotation:
    label: str
    points: list[tuple[int, int]]
    point_labels: list[int]


@dataclass
class PointSegmentationRequest:
    annotations: list[PointAnnotation]
    frame: str


@dataclass
class AnnotationSaveRequest:
    frame_id: int
    annotations: list[dict[str, Any]]  # [{label: str, points: list[tuple[int, int]], point_labels: list[int]}]


async def ensure_inference_running(request: Request) -> tuple[bool, str]:
    """Ensure inference model is loaded and running"""
    if not request.app.is_inference_running or request.app.inference_model is None:
        try:
            request.app.inference_model = FastSAM(FAST_SAM_CHECKPOINT)
            request.app.is_inference_running = True
            return True, ""
        except Exception as e:
            return False, f"Failed to load inference model: {str(e)}"
    return True, ""


@router.get("/", response_class=HTMLResponse)
async def labeling(
    request: Request, 
    calibration_id: int | None = None
):
    context = LabelingContext(request=request)

    # First ensure inference is running
    is_running, error = await ensure_inference_running(request)
    if not is_running:
        return Response(status_code=503, content=f"Error: {error}")

    if calibration_id:
        with Session(engine) as session:
            cal_rec = session.query(CalibrationRecording)\
                .options(
                    joinedload(CalibrationRecording.sim_room)
                    .joinedload(SimRoom.sim_room_classes)
                )\
                .options(
                    joinedload(CalibrationRecording.annotations)
                    .joinedload(Annotation.point_labels)
                )\
                .options(
                    joinedload(CalibrationRecording.annotations)
                    .joinedload(Annotation.sim_room_class)
                )\
                .filter(CalibrationRecording.id == calibration_id)\
                .first()
            
            if cal_rec is None:
                return Response(status_code=404, content="Error: Calibration recording not found")
            
            # Get classes from sim room with IDs and active state
            context.classes = [
                {
                    "label": cls.class_name, 
                    "id": cls.id, 
                    "color": "#00ff00",
                    "active": False
                } 
                for cls in cal_rec.sim_room.sim_room_classes
            ]
            # Set first class as active
            if context.classes:
                context.classes[0]["active"] = True
            
            # Get existing annotations
            context.annotations = []
            for annotation in cal_rec.annotations:
                points = [(label.x, label.y) for label in annotation.point_labels]
                point_labels = [1 if label.label else 0 for label in annotation.point_labels]
                context.annotations.append({
                    "label": annotation.sim_room_class.class_name,
                    "points": points,
                    "point_labels": point_labels
                })

            context.calibration_recording = cal_rec
            context.recording_uuid = cal_rec.recording_uuid
    else:
        return Response(status_code=400, content="Error: No calibration_id provided")

    if is_hx_request(request):
        return templates.TemplateResponse(Template.LABELER, context.to_dict())
    return templates.TemplateResponse(Template.INDEX, context.to_dict())


@router.post("/segmentation", response_class=JSONResponse)
async def get_point_segmentation(request: Request, body: PointSegmentationRequest):
    # read base64 encoded image from request to numpy array
    image = base64_to_numpy(body.frame)
    results = []

    for annotation in body.annotations:
        points = annotation.points
        point_labels = annotation.point_labels
        label = annotation.label

        # get masks for each annotation
        mask = fastsam.get_mask(
            image=image, 
            model=request.app.inference_model, 
            points=points, 
            point_labels=point_labels, 
            conf=0.6, 
            iou=0.8
        )

        results.append({
            "class": label,
            "mask": mask,
            "point_labels": point_labels
        })

    return JSONResponse(content=results)


@router.post("/{calibration_id}/annotations", response_class=JSONResponse)
async def save_annotations(
    request: Request,
    calibration_id: int,
    body: AnnotationSaveRequest
):
    with Session(engine) as session:
        cal_rec = session.query(CalibrationRecording)\
            .options(joinedload(CalibrationRecording.sim_room))\
            .filter(CalibrationRecording.id == calibration_id)\
            .first()
        
        if not cal_rec:
            return Response(status_code=404, content="Calibration recording not found")

        # Delete existing annotations for this frame
        existing_annotations = session.query(Annotation)\
            .filter(Annotation.calibration_recording_id == calibration_id)\
            .all()
        for annotation in existing_annotations:
            session.delete(annotation)

        # Create new annotations
        for annotation_data in body.annotations:
            # Get sim room class by name
            sim_class = session.query(SimRoomClass)\
                .filter(
                    SimRoomClass.sim_room_id == cal_rec.sim_room_id,
                    SimRoomClass.class_name == annotation_data["label"]
                ).first()
            
            if not sim_class:
                continue  # Skip if class no longer exists

            # Create annotation
            annotation = Annotation(
                calibration_recording_id=calibration_id,
                sim_room_class_id=sim_class.id
            )
            session.add(annotation)
            session.flush()  # Get annotation ID

            # Create point labels
            for point, label in zip(annotation_data["points"], annotation_data["point_labels"]):
                point_label = PointLabel(
                    annotation_id=annotation.id,
                    x=point[0],
                    y=point[1],
                    label=(label == 1)
                )
                session.add(point_label)

        session.commit()

    return JSONResponse(content={"success": True})
