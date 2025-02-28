from dataclasses import dataclass

import cv2
from fastapi import APIRouter, Depends, Response
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np
from sqlalchemy.orm import Session, joinedload
from src.api.dependencies import get_labeler, get_selected_class_id
from src.api.models import (
    Labeler,
    LabelingAnnotationsContext,
    LabelingClassesContext,
    LabelingControlsContext,
    Request,
)
from src.config import Template, templates
from src.db import CalibrationRecording, engine
from src.db.models import Annotation, PointLabel, SimRoomClass
from src.utils import encode_to_png, is_hx_request

router = APIRouter(prefix="/labeling")


@router.post("/", response_class=Response)
async def start_labeling(request: Request, calibration_id: int) -> Response:
    with Session(engine) as session:
        cal_rec = (
            session.query(CalibrationRecording)
            .filter(CalibrationRecording.id == calibration_id)
            .first()
        )

        if not cal_rec:
            return Response(
                status_code=404, content="Error: Calibration recording not found"
            )

        try:
            request.app.labeler = Labeler(calibration_recording=cal_rec)
            labeling_context = request.app.labeler.get_labeling_context(request).to_dict()

            if is_hx_request(request):
                return templates.TemplateResponse(Template.LABELER, labeling_context)
            return templates.TemplateResponse(Template.INDEX, labeling_context)
        except Exception as e:
            return Response(status_code=500, content=f"Failed to start labeling: {e!s}")


@router.get("/", response_class=HTMLResponse)
async def labeling(
    request: Request, labeler: Labeler = Depends(get_labeler)
) -> HTMLResponse:
    labeling_context = labeler.get_labeling_context(request).to_dict()
    if is_hx_request(request):
        return templates.TemplateResponse(Template.LABELER, labeling_context)
    return templates.TemplateResponse(Template.INDEX, labeling_context)


@router.get("/point_labels", response_class=JSONResponse)
async def point_labels(labeler: Labeler = Depends(get_labeler)) -> JSONResponse:
    with Session(engine) as session:
        cal_rec_id = labeler.calibration_recording.id
        annotations = (
            session.query(Annotation)
            .filter(
                Annotation.calibration_recording_id == cal_rec_id,
                Annotation.frame_idx == labeler.current_frame_idx,
            )
            .all()
        )

        point_labels = []
        for annotation in annotations:
            for point_label in annotation.point_labels:
                point_label_dict = point_label.to_dict()
                point_label_dict["class_id"] = annotation.sim_room_class_id
                point_labels.append(point_label_dict)

        return JSONResponse(content=point_labels)


@router.get("/current_frame", response_class=Response)
async def current_frame(labeler: Labeler = Depends(get_labeler)) -> Response:
    frame = labeler.get_overlay()
    ret, encoded_png = cv2.imencode(".png", frame)
    if not ret:
        return Response(status_code=500, content="Error: Failed to encode frame")

    return Response(content=encoded_png.tobytes(), media_type="image/png")


@router.get("/controls", response_class=HTMLResponse)
async def controls(
    request: Request,
    polling: bool,
    frame_idx: int | None = None,
    labeler: Labeler = Depends(get_labeler),
    selected_class_id: int = Depends(get_selected_class_id),
) -> HTMLResponse:
    frame_idx = labeler.current_frame_idx if frame_idx is None else frame_idx
    context = LabelingControlsContext(
        request=request,
        current_frame_idx=frame_idx,
        frame_count=labeler.frame_count,
        selected_class_id=selected_class_id,
    )
    
    with Session(engine) as session:
        sim_room_class: SimRoomClass = session.query(SimRoomClass).get(selected_class_id) 
        if sim_room_class:
            context.tracks = labeler.get_tracks()
            context.selected_class_color = sim_room_class.color

    if labeler.tracking_job is not None and labeler.tracking_job.class_id == selected_class_id:
        context.tracking_progress = labeler.tracking_job.progress
        context.is_tracking = True
    
    if not polling:
        labeler.seek(frame_idx)
        context.update_canvas = True

    return templates.TemplateResponse(
        Template.LABELING_CONTROLS, context=context.to_dict()
    )


@router.get("/classes", response_class=HTMLResponse)
async def classes(
    request: Request, selected_class_id: int = -1, labeler: Labeler = Depends(get_labeler)
) -> HTMLResponse:
    with Session(engine) as session:
        classes = (
            session.query(SimRoomClass)
            .filter(SimRoomClass.sim_room_id == labeler.calibration_recording.sim_room_id)
            .all()
        )

        if selected_class_id == -1 and len(classes) > 0:
            selected_class_id = classes[0].id

        labeler.selected_class_id = selected_class_id

        context = LabelingClassesContext(
            request=request,
            selected_class_id=selected_class_id,
            sim_room_id=labeler.calibration_recording.sim_room_id,
            classes=classes,
        )

        return templates.TemplateResponse(
            Template.LABELING_CLASSES, context=context.to_dict()
        )




@router.get("/annotations", response_class=HTMLResponse)
async def annotations(
    request: Request,
    labeler: Labeler = Depends(get_labeler),
    selected_class_id: int = Depends(get_selected_class_id),
) -> HTMLResponse:
    context = LabelingAnnotationsContext(request=request)
    with Session(engine) as session:
        cal_rec_id = labeler.calibration_recording.id
        
        annotations = (
            session.query(Annotation)
            .filter(
                Annotation.calibration_recording_id == cal_rec_id,
                Annotation.sim_room_class_id == selected_class_id,
            )
            .all()
        )

        annotations_dicts = []
        for ann in annotations:
            file = np.load(ann.result_path)
            x1, y1, x2, y2 = file["bbox"]

            frame_crop = labeler.get_frame(ann.frame_idx)[y1:y2, x1:x2]
            encoded_png = encode_to_png(frame_crop)

            annotations_dicts.append(
                {
                    "id": ann.id,
                    "frame_idx": ann.frame_idx,
                    "frame_crop": encoded_png,
                }
            )

        context.annotations = annotations_dicts

    return templates.TemplateResponse(Template.LABELING_ANNOTATIONS, context.to_dict())

@dataclass
class AnnotationPostBody:
    point: tuple[int, int]
    label: int
    delete_point: bool = False

@router.post("/annotations", response_class=Response)
async def post_annotation(
    body: AnnotationPostBody,
    labeler: Labeler = Depends(get_labeler),
    selected_class_id: int = Depends(get_selected_class_id),
) -> Response:
    if selected_class_id == -1:
        return Response(status_code=400, content="Error: No class selected")

    with Session(engine) as session:
        annotation = (
            session.query(Annotation)
            .filter(
                Annotation.calibration_recording_id == labeler.calibration_recording.id,
                Annotation.frame_idx == labeler.current_frame_idx,
                Annotation.sim_room_class_id == selected_class_id,
            )
            .first()
        )

        if body.delete_point and annotation is None:
            return Response(status_code=404, content="Annotation not found")

        if not annotation:
            annotation = Annotation(
                calibration_recording_id=labeler.calibration_recording.id,
                sim_room_class_id=selected_class_id,
                frame_idx=labeler.current_frame_idx,
            )
            session.add(annotation)
            session.flush()

        if body.delete_point:
            closest_point = PointLabel.find_closest(
                annotation.id, body.point[0], body.point[1]
            )
            if not closest_point:
                return Response(status_code=404, content="Point not found")

            session.delete(closest_point)
        else:
            point_label = PointLabel(
                annotation_id=annotation.id,
                x=body.point[0],
                y=body.point[1],
                label=body.label,
            )
            session.add(point_label)

        session.flush()
        points = [(label.x, label.y) for label in annotation.point_labels]
        labels = [int(label.label) for label in annotation.point_labels]

        # delete annotation if no points remain
        if len(points) == 0:
            session.delete(annotation)
            session.commit()
            return await current_frame(labeler)

        # update annotation mask and bounding box
        try:
            labeler.predict_image(points, labels)
        except ValueError:
            return Response(status_code=500, content="Error: Failed to predict image")
        
        session.commit()
        return await current_frame(labeler)


@router.delete("/annotations/{annotation_id}", response_class=HTMLResponse)
async def delete_calibration_annotation(
    request: Request, annotation_id: int, labeler: Labeler = Depends(get_labeler)
) -> HTMLResponse:
    with Session(engine) as session:
        annotation = (
            session.query(Annotation).filter(Annotation.id == annotation_id).first()
        )

        if not annotation:
            return HTMLResponse(status_code=404, content="Annotation not found")

        sim_room_class_id = annotation.sim_room_class_id
        session.delete(annotation)
        session.commit()

        return await annotations(request, labeler, sim_room_class_id)
    
@router.post("/tracking", response_class=HTMLResponse)
async def tracking(request: Request, labeler: Labeler = Depends(get_labeler), selected_class_id: int = Depends(get_selected_class_id)) -> HTMLResponse:
    if labeler.tracking_job is not None:
        return HTMLResponse(status_code=400, content="Error: Tracking already in progress")

    if selected_class_id == -1:
        return HTMLResponse(status_code=400, content="Error: No class selected")

    with Session(engine) as session:
        annotations = (
            session.query(Annotation)
            .filter(
                Annotation.calibration_recording_id == labeler.calibration_recording.id,
                Annotation.sim_room_class_id == selected_class_id,
            )
            .options(
                joinedload(Annotation.point_labels), 
            )
            .all()
        )

        labeler.create_tracking_job(annotations)

    return await controls(
        request, 
        polling=False, 
        frame_idx=labeler.current_frame_idx, 
        labeler=labeler, 
        selected_class_id=selected_class_id
    )