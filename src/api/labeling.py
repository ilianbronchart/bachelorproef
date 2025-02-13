from fastapi import APIRouter, Response
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from src.api.models import LabelingContext, Request
from src.config import FAST_SAM_CHECKPOINT, Template, templates
from src.db import CalibrationRecording, engine
from src.utils import is_hx_request
from ultralytics import FastSAM

router = APIRouter(prefix="/labeling")


@router.post("/", response_class=JSONResponse)
async def start_labeling(request: Request, calibration_id: int):
    with Session(engine) as session:
        cal_rec = session.query(CalibrationRecording).filter(CalibrationRecording.id == calibration_id).first()

        if not cal_rec:
            return Response(status_code=404, content="Error: Calibration recording not found")

        try:
            request.app.labeling_context = LabelingContext(
                request=request,
                calibration_recording=cal_rec,
                sim_room=cal_rec.sim_room,
                recording=cal_rec.recording,
                model=FastSAM(FAST_SAM_CHECKPOINT),
                classes=cal_rec.sim_room.classes,
                annotations=cal_rec.annotations,
            )
        except Exception as e:
            return Response(status_code=500, content=f"Failed to start labeling: {e!s}")

    if is_hx_request(request):
        return templates.TemplateResponse(Template.LABELER, request.app.labeling_context)
    return templates.TemplateResponse(Template.INDEX, request.app.labeling_context)


@router.get("/", response_class=HTMLResponse)
async def labeling(request: Request, calibration_id: int):
    if not request.app.labeling_context:
        return Response(status_code=503, content="Error: Inference model not running")

    if request.app.labeling_context.calibration_recording.id != calibration_id:
        return Response(status_code=400, content="Error: Calibration recording mismatch")

    with Session(engine) as session:
        cal_rec = session.query(CalibrationRecording).filter(CalibrationRecording.id == calibration_id).first()

        if not cal_rec:
            return Response(status_code=404, content="Error: Calibration recording not found")

        request.app.labeling_context.classes = cal_rec.sim_room.classes
        request.app.labeling_context.annotations = cal_rec.annotations

    if is_hx_request(request):
        return templates.TemplateResponse(Template.LABELER, context.to_dict())
    return templates.TemplateResponse(Template.INDEX, context.to_dict())
