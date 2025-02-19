from dataclasses import dataclass
from fastapi import APIRouter, Form, Response
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from src.api.models import Request, SimRoomsContext
from src.api.models.context import ClassListContext
from src.config import Template, templates
from src.db import Recording, SimRoom, SimRoomClass
from src.db.db import engine
from src.db.models import CalibrationRecording
from src.utils import is_hx_request
from src.db.models.calibration import CalibrationRecording, Annotation, PointLabel  # added import
from fastapi import Request

router = APIRouter(prefix="/simrooms")

@dataclass
class AnnotationBody:
    calibration_id: int

    points: list[tuple[int, int]]
    """Segmentation Points: List of points (x, y)."""

    labels: list[int]
    """Segmentation Point Labels: 1 for positive, 0 for negative."""

    mask: str
    """Base64 encoded mask image."""

    frame_crop: str
    """Base64 encoded frame crop image."""

    bounding_box: list[int]
    """Bounding box coordinates: x1, y1, x2, y2."""

    sim_room_class_id: int
    """Sim Room Class ID."""

    frame_idx: int


@router.get("/", response_class=HTMLResponse)
async def simrooms(request: Request, sim_room_id: int | None = None):
    context = SimRoomsContext(
        request=request,
        recordings=Recording.get_all(),
    )

    with Session(engine) as session:
        context.sim_rooms = session.query(SimRoom).all()

        if sim_room_id:
            context.selected_sim_room = session.query(SimRoom).get(sim_room_id)

            if not context.selected_sim_room:
                headers = {"HX-Push-Url": "/simrooms"}
                return Response(status_code=404, content="Sim Room not found", headers=headers)

            context.calibration_recordings = context.selected_sim_room.calibration_recordings
            context.classes = context.selected_sim_room.classes

        headers = {"HX-Push-Url": f"/simrooms/?sim_room_id={sim_room_id}"}
        if is_hx_request(request):
            return templates.TemplateResponse(Template.SIMROOMS, context.to_dict(), headers=headers)
        return templates.TemplateResponse(Template.INDEX, context.to_dict(), headers=headers)


@router.post("/add", response_class=HTMLResponse)
async def add_sim_room(request: Request, name: str = Form(...)):
    try:
        with Session(engine) as session:
            sim_room = SimRoom(name=name)
            session.add(sim_room)
            session.commit()
            session.refresh(sim_room)

        return await simrooms(request, sim_room_id=sim_room.id)
    except Exception as e:
        return Response(status_code=500, content=f"Error: {e!s}")


@router.delete("/{sim_room_id}", response_class=HTMLResponse)
async def delete_sim_room(request: Request, sim_room_id: int):
    try:
        with Session(engine) as session:
            sim_room = session.query(SimRoom).get(sim_room_id)
            if not sim_room:
                return Response(status_code=404, content="Sim Room not found")
            session.delete(sim_room)
            session.commit()

            response = await simrooms(request)
            response.headers["HX-Push-Url"] = "/simrooms"
            return response
    except Exception as e:
        return Response(status_code=500, content=f"Error: {e!s}")


@router.post("/{sim_room_id}/classes/add", response_class=HTMLResponse)
async def add_sim_room_class(
    request: Request,
    sim_room_id: int,
    class_name: str = Form(...),
):
    try:
        with Session(engine) as session:
            sim_room = session.query(SimRoom).get(sim_room_id)
            if not sim_room:
                return Response(status_code=404, content="Sim Room not found")

            sim_room_class = SimRoomClass(sim_room_id=sim_room.id, class_name=class_name)
            session.add(sim_room_class)
            session.commit()

            context = ClassListContext(
                request=request,
                selected_sim_room=sim_room,
                classes=sim_room.classes,
            )
            return templates.TemplateResponse(Template.CLASS_LIST, context.to_dict())
    except Exception as e:
        return Response(status_code=500, content=f"Error: {e!s}")


@router.delete("/{sim_room_id}/classes/{class_id}", response_class=HTMLResponse)
async def delete_sim_room_class(request: Request, sim_room_id: int, class_id: int):
    try:
        with Session(engine) as session:
            sim_room = session.query(SimRoom).get(sim_room_id)
            if not sim_room:
                return Response(status_code=404, content="Sim Room not found")

            sim_room_class = session.query(SimRoomClass).get(class_id)
            if not sim_room_class:
                return Response(status_code=404, content="Sim Room Class not found")

            session.delete(sim_room_class)
            session.commit()

            context = ClassListContext(
                request=request,
                selected_sim_room=sim_room,
                classes=sim_room.classes,
            )
            return templates.TemplateResponse(Template.CLASS_LIST, context.to_dict())
    except Exception as e:
        return Response(status_code=500, content=f"Error: {e!s}")


@router.post("/{sim_room_id}/calibration_recordings", response_class=HTMLResponse)
async def add_calibration_recording(request: Request, sim_room_id: int, recording_uuid: str = Form(...)):
    try:
        with Session(engine) as session:
            sim_room = session.query(SimRoom).get(sim_room_id)

            if not sim_room:
                return Response(status_code=404, content="Sim Room not found")

            if any(cr.recording_uuid == recording_uuid for cr in sim_room.calibration_recordings):
                return Response(status_code=400, content="Calibration Recording already exists in this Sim Room")

            recording = session.query(Recording).get(recording_uuid)
            if not sim_room or not recording:
                return Response(status_code=404, content="Sim Room or Recording not found")

            session.add(CalibrationRecording(sim_room_id=sim_room_id, recording_uuid=recording_uuid))
            session.commit()

            return await simrooms(request, sim_room_id=sim_room_id)
    except Exception as e:
        return Response(status_code=500, content=str(e))


@router.delete("/{sim_room_id}/calibration_recordings/{calibration_id}", response_class=HTMLResponse)
async def delete_calibration_recording(request: Request, sim_room_id: int, calibration_id: int):
    try:
        with Session(engine) as session:
            cal_rec = session.query(CalibrationRecording).get(calibration_id)
            if not cal_rec:
                return Response(status_code=404, content="Calibration Recording not found")
            session.delete(cal_rec)
            session.commit()

            return await simrooms(request, sim_room_id=sim_room_id)
    except Exception as e:
        return Response(status_code=500, content=f"Error: {e!s}")

    return Response(status_code=200)

@router.post("/{sim_room_id}/calibration_recordings/{calibration_id}/annotations", response_class=JSONResponse)
async def add_calibration_annotation(request: Request, body: AnnotationBody):
    if body.bounding_box and len(body.bounding_box) != 4:
        return Response(status_code=400, content="Bounding box must have x1, y1, x2, y2 coordinates")

    with Session(engine) as session:
        # Fetch the calibration recording
        cal_rec = session.query(CalibrationRecording).filter(CalibrationRecording.id == body.calibration_id).first()
        if not cal_rec:
            return Response(status_code=404, content="Calibration recording not found")
        
        # Ensure the sim room relation is loaded
        if not cal_rec.sim_room or not cal_rec.sim_room.classes:
            return Response(status_code=400, content="No classes found for the associated Sim Room")
        
        # Check for an existing annotation for this frame and class
        annotation = (
            session.query(Annotation)
            .filter(
            Annotation.calibration_recording_id == cal_rec.id,
            Annotation.frame_idx == body.frame_idx,
            Annotation.sim_room_class_id == body.sim_room_class_id
            )
            .first()
        )
        bounding_box = ",".join(map(str, body.bounding_box))
        
        if annotation:
            # Update the annotation mask and clear existing point labels
            annotation.annotation_mask = body.mask
            annotation.bounding_box = bounding_box
            annotation.frame_crop = body.frame_crop
            for pl in annotation.point_labels:
                session.delete(pl)
        else:
            annotation = Annotation(
                calibration_recording_id=cal_rec.id,
                sim_room_class_id=body.sim_room_class_id,
                frame_idx=body.frame_idx,
                annotation_mask=body.mask,
                bounding_box=bounding_box,
                frame_crop=body.frame_crop,
            )
            session.add(annotation)
        
        # Add new point labels
        for (x, y), label in zip(body.points, body.labels):
            pl = PointLabel(annotation=annotation, x=x, y=y, label=bool(label))
            session.add(pl)
        
        session.commit()
        return JSONResponse({"status": "success"})


@router.delete("/{sim_room_id}/calibration_recordings/{calibration_id}/annotations/{annotation_id}", response_class=JSONResponse)
async def delete_calibration_annotation(sim_room_id: int, calibration_id: int, annotation_id: int):
    with Session(engine) as session:
        if not session.query(SimRoom).get(sim_room_id):
            return Response(status_code=404, content="Sim Room not found")
        
        if not session.query(CalibrationRecording).get(calibration_id):
            return Response(status_code=404, content="Calibration Recording not found")

        annotation = session.query(Annotation).filter(
            Annotation.id == annotation_id,
            Annotation.calibration_recording_id == calibration_id
        ).first()
        if not annotation:
            return Response(status_code=404, content="Annotation not found")
        session.delete(annotation)
        session.commit()
        return JSONResponse({"status": "success"})