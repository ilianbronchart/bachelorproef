from fastapi import APIRouter, Form, Response
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from src.api.models import Request
from src.db import CalibrationRecording, Recording, SimRoom
from src.db.db import engine

router = APIRouter(prefix="/calibration_recordings")


@router.post("/{sim_room_id}", response_class=HTMLResponse)
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
        return Response(status_code=500, content=str(e))

    return Response(status_code=200)


@router.delete("/{calibration_id}", response_class=HTMLResponse)
async def delete_calibration_recording(request: Request, calibration_id: int):
    try:
        with Session(engine) as session:
            cal_rec = session.query(CalibrationRecording).get(calibration_id)
            if not cal_rec:
                return Response(status_code=404, content="Calibration Recording not found")
            session.delete(cal_rec)
            session.commit()
    except Exception as e:
        return Response(status_code=500, content=f"Error: {e!s}")

    return Response(status_code=200)


@router.get("/{calibration_id}/annotations", response_class=JSONResponse)
async def get_calibration_annotations(request: Request, calibration_id: int):
    try:
        with Session(engine) as session:
            cal_rec = session.query(CalibrationRecording).get(calibration_id)
            if not cal_rec:
                return JSONResponse(status_code=404, content={"error": "Calibration Recording not found"})
            return JSONResponse(content=[annotation.to_dict() for annotation in cal_rec.annotations])
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
