from dataclasses import dataclass, field

from fastapi import APIRouter, Form, Response
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from src.config import BaseContext, Request, Template, templates
from src.core.utils import is_hx_request
from src.db import CalibrationRecording, Recording, SimRoom, SimRoomClass
from src.db.db import engine

router = APIRouter(prefix="/simrooms")


@dataclass
class SimRoomsContext(BaseContext):
    # Context for populating the three columns on the page.
    sim_rooms: list[SimRoom] = field(default_factory=list)
    local_recordings: list[dict] = field(default_factory=list)
    calibration_recordings: list[CalibrationRecording] = field(default_factory=list)
    selected_sim_room: SimRoom | None = None
    classes: list[dict] = field(default_factory=list)
    error_msg: str | None = None
    content: str = Template.SIMROOMS


def build_simrooms_context(
    request: Request, selected_sim_room_id: int | None = None, error_msg: str | None = None
) -> SimRoomsContext:
    context = SimRoomsContext(request=request)

    with Session(engine) as session:
        context.sim_rooms = session.query(SimRoom).all()
        context.local_recordings = [rec.to_dict() for rec in session.query(Recording).all()]

        if selected_sim_room_id is not None:
            calibration_recordings = (
                session.query(CalibrationRecording)
                .filter(CalibrationRecording.sim_room_id == selected_sim_room_id)
                .all()
            )
            context.calibration_recordings = [rec.to_dict() for rec in calibration_recordings]

            sim_room = session.query(SimRoom).get(selected_sim_room_id)
            if sim_room:
                context.selected_sim_room = sim_room
                context.classes = [cls_.to_dict() for cls_ in sim_room.classes]
        else:
            context.calibration_recordings = []

    context.error_msg = error_msg
    return context


@router.get("/", response_class=HTMLResponse)
async def simrooms(request: Request, sim_room_id: int | None = None):
    context = build_simrooms_context(request, selected_sim_room_id=sim_room_id)
    if is_hx_request(request):
        return templates.TemplateResponse(Template.SIMROOMS, context.to_dict())
    return templates.TemplateResponse(Template.INDEX, context.to_dict())


@router.post("/add", response_class=HTMLResponse)
async def add_sim_room(request: Request, name: str = Form(...)):
    try:
        with Session(engine) as session:
            new_sim_room = SimRoom(name=name)
            session.add(new_sim_room)
            session.flush()  # Flush to get the new ID
            new_sim_room_id = new_sim_room.id
            session.commit()
    except Exception as e:
        context = build_simrooms_context(request, error_msg=str(e))
        return templates.TemplateResponse(Template.SIMROOMS, context.to_dict())

    context = build_simrooms_context(request, selected_sim_room_id=new_sim_room_id)
    return templates.TemplateResponse(Template.SIMROOMS, context.to_dict())


@router.delete("/{sim_room_id}", response_class=HTMLResponse)
async def delete_sim_room(request: Request, sim_room_id: int):
    try:
        with Session(engine) as session:
            sim_room = session.query(SimRoom).get(sim_room_id)
            if not sim_room:
                return Response(status_code=404, content="Sim Room not found")
            session.delete(sim_room)
            session.commit()
    except Exception as e:
        return Response(status_code=500, content=f"Error: {e!s}")

    context = build_simrooms_context(request)
    return templates.TemplateResponse(Template.SIMROOMS, context.to_dict())


@router.post("/{sim_room_id}/classes/add", response_class=HTMLResponse)
async def add_sim_room_class(
    request: Request,
    sim_room_id: int,
    class_name: str = Form(...),
    color: str = Form(...),
):
    try:
        with Session(engine) as session:
            sim_room = session.query(SimRoom).get(sim_room_id)
            if not sim_room:
                return Response(status_code=404, content="Sim Room not found")
            new_class = SimRoomClass(
                sim_room_id=sim_room.id,
                class_name=class_name,
                color=color,
            )
            session.add(new_class)
            session.commit()
            session.refresh(new_class)  # Refresh to get the ID
            # Get updated classes
            classes = [cls.to_dict() for cls in sim_room.classes]
    except Exception as e:
        return Response(status_code=500, content=f"Error: {str(e)}")

    # Return just the class list component
    return templates.TemplateResponse(
        "components/class-list.jinja",
        {"request": request, "sim_room_id": sim_room_id, "classes": classes}
    )


@router.delete("/{sim_room_id}/classes/{class_id}", response_class=HTMLResponse)
async def delete_sim_room_class(request: Request, sim_room_id: int, class_id: int):
    try:
        with Session(engine) as session:
            sim_class = session.query(SimRoomClass).get(class_id)
            if not sim_class:
                return Response(status_code=404, content="Class not found")
            session.delete(sim_class)
            session.commit()
            # Get updated classes
            sim_room = session.query(SimRoom).get(sim_room_id)
            classes = [cls.to_dict() for cls in sim_room.classes]
    except Exception as e:
        return Response(status_code=500, content=f"Error: {str(e)}")

    # Return just the class list component
    return templates.TemplateResponse(
        "components/class-list.jinja",
        {"request": request, "sim_room_id": sim_room_id, "classes": classes}
    )


@router.post("/{sim_room_id}/calibration_recordings", response_class=HTMLResponse)
async def add_calibration_recording(request: Request, sim_room_id: int, recording_uuid: str = Form(...)):
    try:
        with Session(engine) as session:
            sim_room = session.query(SimRoom).get(sim_room_id)
            recording = session.query(Recording).get(recording_uuid)
            if not sim_room or not recording:
                return Response(status_code=404, content="Sim Room or Recording not found")

            session.add(CalibrationRecording(sim_room_id=sim_room.id, recording_uuid=recording.uuid))
            session.commit()
    except Exception as e:
        context = build_simrooms_context(request, error_msg=str(e))
        return templates.TemplateResponse(Template.SIMROOMS, context.to_dict())

    headers = {"HX-Push-Url": f"/simrooms/?sim_room_id={sim_room_id}"}
    context = build_simrooms_context(request, selected_sim_room_id=sim_room_id)
    return templates.TemplateResponse(Template.SIMROOMS, context.to_dict(), headers=headers)


@router.delete("/{sim_room_id}/calibration_recordings/{calibration_id}", response_class=HTMLResponse)
async def delete_calibration_recording(request: Request, sim_room_id: int, calibration_id: int):
    try:
        with Session(engine) as session:
            cal_rec = session.query(CalibrationRecording).get(calibration_id)
            if not cal_rec:
                return Response(status_code=404, content="Calibration Recording not found")
            session.delete(cal_rec)
            session.commit()
    except Exception as e:
        return Response(status_code=500, content=f"Error: {e!s}")

    headers = {"HX-Push-Url": f"/simrooms/?sim_room_id={sim_room_id}"}
    context = build_simrooms_context(request, selected_sim_room_id=sim_room_id)
    return templates.TemplateResponse(Template.SIMROOMS, context.to_dict(), headers=headers)


@router.put("/{sim_room_id}/classes/{class_id}/color", response_class=JSONResponse)
async def update_class_color(request: Request, sim_room_id: int, class_id: int, color: str = Form(...)):
    try:
        with Session(engine) as session:
            sim_class = session.query(SimRoomClass).get(class_id)
            if not sim_class:
                return Response(status_code=404, content="Class not found")
            sim_class.color = color
            session.commit()
            # Return the updated class data
            return JSONResponse(content=sim_class.to_dict())
    except Exception as e:
        return Response(status_code=500, content=f"Error: {str(e)}")
