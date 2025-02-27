from fastapi import APIRouter, Form, Response
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from src.api.models import Request, SimRoomsContext
from src.api.models.context import ClassListContext
from src.config import Template, templates
from src.db import Recording, SimRoom, SimRoomClass
from src.db.db import engine
from src.db.models import CalibrationRecording
from src.utils import is_hx_request

router = APIRouter(prefix="/simrooms")


@router.get("/", response_class=HTMLResponse)
async def simrooms(request: Request, sim_room_id: int | None = None) -> HTMLResponse | Response:
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
async def add_sim_room(request: Request, name: str = Form(...)) -> HTMLResponse | Response:
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
async def delete_sim_room(request: Request, sim_room_id: int) -> HTMLResponse | Response:
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


@router.get("/{sim_room_id}/classes", response_class=HTMLResponse)
async def sim_room_classes(request: Request, sim_room_id: int) -> HTMLResponse | Response:
    try:
        with Session(engine) as session:
            sim_room = session.query(SimRoom).get(sim_room_id)
            if not sim_room:
                return Response(status_code=404, content="Sim Room not found")

            context = ClassListContext(
                request=request,
                selected_sim_room=sim_room,
                classes=sim_room.classes,
            )
            return templates.TemplateResponse(Template.CLASS_LIST, context.to_dict())
    except Exception as e:
        return Response(status_code=500, content=f"Error: {e!s}")


@router.post("/{sim_room_id}/classes/add", response_class=HTMLResponse)
async def add_sim_room_class(
    request: Request,
    sim_room_id: int,
    class_name: str = Form(...),
) -> HTMLResponse | Response:
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
async def delete_sim_room_class(request: Request, sim_room_id: int, class_id: int) -> HTMLResponse | Response:
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
async def add_calibration_recording(
    request: Request, sim_room_id: int, recording_uuid: str = Form(...)
) -> HTMLResponse | Response:
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


@router.delete("/{sim_room_id}/calibration_recordings/{calibration_id}", response_class=Response)
async def delete_calibration_recording(request: Request, sim_room_id: int, calibration_id: int) -> Response:
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
