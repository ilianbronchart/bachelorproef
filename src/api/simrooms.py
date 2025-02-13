from typing import List
from fastapi import APIRouter, Form, Response
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from src.api.models import Request, SimRoomsContext
from src.config import Template, templates
from src.db import Recording, SimRoom, SimRoomClass
from src.db.db import engine
from src.utils import is_hx_request

router = APIRouter(prefix="/simrooms")

@router.get("/", response_class=HTMLResponse)
async def simrooms(request: Request, sim_room_id: int | None = None):
    context = SimRoomsContext(
        request=request,
        recordings=Recording.get_all(),
    )

    with Session(engine) as session:
        context.sim_rooms = session.query(SimRoom).all()
        context.selected_sim_room = session.query(SimRoom).get(sim_room_id)

        if context.selected_sim_room:
            context.calibration_recordings = context.selected_sim_room.calibration_recordings
            context.classes = context.selected_sim_room.classes

    if is_hx_request(request):
        return templates.TemplateResponse(Template.SIMROOMS, context.to_dict())
    return templates.TemplateResponse(Template.INDEX, context.to_dict())


@router.post("/add", response_class=HTMLResponse)
async def add_sim_room(request: Request, name: str = Form(...)):
    try:
        with Session(engine) as session:
            sim_room = SimRoom(name=name)
            session.add(sim_room)
            session.commit()
            session.refresh(sim_room)
            
            response = await simrooms(request, sim_room_id=sim_room.id)
            response.headers["HX-Push-Url"] = f"/simrooms/?sim_room_id={sim_room.id}"
            return response
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
    except Exception as e:
        return Response(status_code=500, content=f"Error: {e!s}")

    headers = {"HX-Push-Url": "/simrooms/"}
    context = build_simrooms_context(request)
    return templates.TemplateResponse(Template.SIMROOMS, context.to_dict(), headers=headers)


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
    except Exception as e:
        context = build_simrooms_context(request, error_msg=str(e))
        return templates.TemplateResponse(Template.SIMROOMS, context.to_dict())

    headers = {"HX-Push-Url": f"/simrooms/?sim_room_id={sim_room_id}"}
    context = build_simrooms_context(request, selected_sim_room_id=sim_room_id)
    return templates.TemplateResponse(Template.SIMROOMS, context.to_dict(), headers=headers)


@router.delete("/{sim_room_id}/classes/{class_id}", response_class=HTMLResponse)
async def delete_sim_room_class(request: Request, sim_room_id: int, class_id: int):
    try:
        with Session(engine) as session:
            sim_room_class = session.query(SimRoomClass).get(class_id)
            if not sim_room_class:
                return Response(status_code=404, content="Sim Room Class not found")
            session.delete(sim_room_class)
            session.commit()
    except Exception as e:
        return Response(status_code=500, content=f"Error: {e!s}")

    headers = {"HX-Push-Url": f"/simrooms/?sim_room_id={sim_room_id}"}
    context = build_simrooms_context(request, selected_sim_room_id=sim_room_id)
    return templates.TemplateResponse(Template.SIMROOMS, context.to_dict(), headers=headers)
