from dataclasses import dataclass, field
from typing import Any, List, Optional

from fastapi import APIRouter, Form, Response
from fastapi.responses import HTMLResponse
from src.config import BaseContext, Request, Template, templates
from src.core.utils import is_hx_request
from src.db import Recording, CalibrationRecording, SimRoom, SimRoomClass
import base64
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Union

from g3pylib.recordings.recording import Recording as GlassesRecording
from sqlalchemy import Column, String
from sqlalchemy.orm import Session, relationship, joinedload

from src.config import RECORDINGS_PATH
from src.core.utils import download_file
from src.db.db import Base, engine


router = APIRouter(prefix="/simrooms")


@dataclass
class SimRoomsContext(BaseContext):
    # Context for populating the three columns on the page.
    sim_rooms: List[SimRoom] = field(default_factory=list)
    local_recordings: List[dict[str, Any]] = field(default_factory=list)
    calibration_recordings: List[CalibrationRecording] = field(default_factory=list)
    selected_sim_room: Optional[SimRoom] = None
    sim_room_classes: List[SimRoomClass] = field(default_factory=list)
    error_msg: Optional[str] = None
    content: str = Template.SIMROOMS


def build_simrooms_context(
    request: Request,
    selected_sim_room_id: Optional[int] = None,
    error_msg: Optional[str] = None
) -> SimRoomsContext:
    """
    Helper function to build the SimRoomsContext.
    """
    context = SimRoomsContext(request=request)
    context.sim_rooms = SimRoom.get_all()
    context.local_recordings = [rec.get_formatted() for rec in Recording.get_all()]
    
    # Only load calibration recordings if a sim room is selected
    if selected_sim_room_id is not None:
        with Session(engine) as session:
            calibration_recordings = session.query(CalibrationRecording)\
                .filter(CalibrationRecording.sim_room_id == selected_sim_room_id)\
                .options(joinedload(CalibrationRecording.recording))\
                .all()
            context.calibration_recordings = [rec.get_formatted() for rec in calibration_recordings]
            
            sim_room = session.query(SimRoom)\
                .options(joinedload(SimRoom.sim_room_classes))\
                .get(selected_sim_room_id)
            if sim_room:
                context.selected_sim_room = sim_room
                context.sim_room_classes = sim_room.sim_room_classes
    else:
        context.calibration_recordings = []
        
    context.error_msg = error_msg
    return context


@router.get("/", response_class=HTMLResponse)
async def simrooms(request: Request, sim_room_id: Optional[int] = None):
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
        context = build_simrooms_context(request, error_msg="Failed to create Sim Room, please try again.")
        return templates.TemplateResponse(Template.SIMROOMS, context.to_dict())

    context = build_simrooms_context(request, selected_sim_room_id=new_sim_room_id)
    return templates.TemplateResponse(Template.SIMROOMS, context.to_dict())


@router.delete("/{sim_room_id}", response_class=HTMLResponse)
async def delete_sim_room(request: Request, sim_room_id: int):
    sim_room = SimRoom.get(sim_room_id)
    if not sim_room:
        return Response(status_code=404, content="Sim Room not found")
    try:
        with Session(engine) as session:
            session.delete(sim_room)
            session.commit()
    except Exception as e:
        return Response(status_code=500, content=f"Error: {str(e)}")
    context = build_simrooms_context(request)
    return templates.TemplateResponse(Template.SIMROOMS, context.to_dict())


@router.post("/{sim_room_id}/classes/add", response_class=HTMLResponse)
async def add_sim_room_class(
    request: Request,
    sim_room_id: int,
    class_name: str = Form(...)
):
    sim_room = SimRoom.get(sim_room_id)
    if not sim_room:
        return Response(status_code=404, content="Sim Room not found")
    try:
        with Session(engine) as session:
            session.add(SimRoomClass(sim_room_id=sim_room.id, class_name=class_name))
            session.commit()
    except Exception as e:
        context = build_simrooms_context(request, selected_sim_room_id=sim_room_id, error_msg=str(e))
        return templates.TemplateResponse(Template.SIMROOMS, context.to_dict())
    context = build_simrooms_context(request, selected_sim_room_id=sim_room_id)
    return templates.TemplateResponse(Template.SIMROOMS, context.to_dict())


@router.delete("/{sim_room_id}/classes/{class_id}", response_class=HTMLResponse)
async def delete_sim_room_class(
    request: Request, sim_room_id: int, class_id: int
):
    sim_room = SimRoom.get(sim_room_id)
    if not sim_room:
        return Response(status_code=404, content="Sim Room not found")
    sim_class = SimRoomClass.get(class_id)
    if not sim_class:
        return Response(status_code=404, content="Class not found")
    try:
        with Session(engine) as session:
            session.delete(sim_class)
            session.commit()
    except Exception as e:
        return Response(status_code=500, content=  f"Error: {str(e)}")
    context = build_simrooms_context(request, selected_sim_room_id=sim_room_id)
    return templates.TemplateResponse(Template.SIMROOMS, context.to_dict())


@router.post("/{sim_room_id}/calibration_recordings", response_class=HTMLResponse)
async def add_calibration_recording(request: Request,
                                    sim_room_id: int,
                                    recording_uuid: str = Form(...)):
    sim_room = SimRoom.get(sim_room_id)
    recording = Recording.get(recording_uuid)
    if not sim_room or not recording:
        return Response(status_code=404, content="Sim Room or Recording not found")
    try:
        with Session(engine) as session:
            session.add(CalibrationRecording(sim_room_id=sim_room.id, recording_uuid=recording.uuid))
            session.commit()
    except Exception as e:
        context = build_simrooms_context(request, error_msg=str(e))
        return templates.TemplateResponse(Template.SIMROOMS, context.to_dict())
    context = build_simrooms_context(request, selected_sim_room_id=sim_room_id)
    return templates.TemplateResponse(Template.SIMROOMS, context.to_dict())


@router.delete("/{sim_room_id}/calibration_recordings/{calibration_id}", response_class=HTMLResponse)
async def delete_calibration_recording(
    request: Request,
    sim_room_id: int,
    calibration_id: int
):
    try:
        with Session(engine) as session:
            cal_rec = session.query(CalibrationRecording).get(calibration_id)
            if not cal_rec:
                return Response(status_code=404, content="Calibration Recording not found")
            session.delete(cal_rec)
            session.commit()
    except Exception as e:
        return Response(status_code=500, content=f"Error: {str(e)}")
    
    context = build_simrooms_context(request, selected_sim_room_id=sim_room_id)
    return templates.TemplateResponse(Template.SIMROOMS, context.to_dict())
