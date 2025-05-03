from fastapi import APIRouter, Depends, Form
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.exceptions import NotFoundError
from src.api.models.context import ClassListContext, Request, SimRoomsContext
from src.api.repositories import simrooms_repo
from src.api.services import recordings_service
from src.config import Template, templates
from src.utils import is_hx_request

router = APIRouter(prefix="/simrooms")


@router.get("/", response_class=HTMLResponse)
async def simrooms(
    request: Request, sim_room_id: int | None = None, db: Session = Depends(get_db)
) -> HTMLResponse:
    recordings = recordings_service.get_all(db)
    sim_rooms = simrooms_repo.get_all_simrooms(db)
    context = SimRoomsContext(
        _request=request, recordings=recordings, sim_rooms=sim_rooms
    )

    if sim_room_id:
        try:
            sim_room = simrooms_repo.get_simroom(db, sim_room_id)
        except NotFoundError:
            headers = {"HX-Push-Url": "/simrooms"}
            return HTMLResponse(
                status_code=404, content="Sim Room not found", headers=headers
            )

        sim_room.calibration_recordings.sort(key=lambda cr: cr.recording.created)
        context.selected_sim_room = sim_room

    headers = {"HX-Push-Url": f"/simrooms/?sim_room_id={sim_room_id}"}
    if is_hx_request(request):
        return templates.TemplateResponse(
            Template.SIMROOMS, context.model_dump(), headers=headers
        )
    return templates.TemplateResponse(
        Template.INDEX, context.model_dump(), headers=headers
    )


@router.post("/add", response_class=HTMLResponse)
async def add_sim_room(
    request: Request, name: str = Form(...), db: Session = Depends(get_db)
) -> HTMLResponse:
    sim_room = simrooms_repo.create_simroom(db, name=name)
    response = await simrooms(request, sim_room_id=sim_room.id, db=db)
    response.headers["HX-Push-Url"] = "/simrooms"
    return response


@router.delete("/{sim_room_id}", response_class=HTMLResponse)
async def delete_sim_room(
    request: Request, sim_room_id: int, db: Session = Depends(get_db)
) -> HTMLResponse:
    simrooms_repo.delete_simroom(db, sim_room_id)
    response = await simrooms(request, db=db)
    response.headers["HX-Push-Url"] = "/simrooms"
    return response


@router.get("/{sim_room_id}/classes", response_class=HTMLResponse)
async def sim_room_classes(
    request: Request, sim_room_id: int, db: Session = Depends(get_db)
) -> HTMLResponse:
    sim_room = simrooms_repo.get_simroom(db, sim_room_id)
    context = ClassListContext(
        _request=request,
        selected_sim_room=sim_room,
    )
    return templates.TemplateResponse(Template.CLASS_LIST, context.model_dump())


@router.post("/{sim_room_id}/classes/add", response_class=HTMLResponse)
async def add_sim_room_class(
    request: Request,
    sim_room_id: int,
    class_name: str = Form(...),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    simrooms_repo.create_simroom_class(
        db,
        sim_room_id=sim_room_id,
        class_name=class_name,
    )
    sim_room = simrooms_repo.get_simroom(db, sim_room_id)

    context = ClassListContext(
        _request=request,
        selected_sim_room=sim_room,
    )
    return templates.TemplateResponse(Template.CLASS_LIST, context.model_dump())


@router.delete("/{sim_room_id}/classes/{class_id}", response_class=HTMLResponse)
async def delete_sim_room_class(
    request: Request, sim_room_id: int, class_id: int, db: Session = Depends(get_db)
) -> HTMLResponse:
    simrooms_repo.delete_simroom_class(db, class_id)
    sim_room = simrooms_repo.get_simroom(db, sim_room_id)
    context = ClassListContext(
        _request=request,
        selected_sim_room=sim_room,
    )
    return templates.TemplateResponse(Template.CLASS_LIST, context.model_dump())


@router.post("/{sim_room_id}/calibration_recordings", response_class=HTMLResponse)
async def add_calibration_recording(
    request: Request,
    sim_room_id: int,
    recording_id: str = Form(...),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    simrooms_repo.add_calibration_recording(
        db, sim_room_id=sim_room_id, recording_id=recording_id
    )
    return await simrooms(request, sim_room_id=sim_room_id, db=db)


@router.delete(
    "/{sim_room_id}/calibration_recordings/{calibration_id}", response_class=HTMLResponse
)
async def delete_calibration_recording(
    request: Request, sim_room_id: int, calibration_id: int, db: Session = Depends(get_db)
) -> HTMLResponse:
    simrooms_repo.delete_calibration_recording(db, calibration_id)
    return await simrooms(request, sim_room_id=sim_room_id, db=db)
