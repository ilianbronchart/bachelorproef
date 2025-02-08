from dataclasses import dataclass, field

from fastapi import APIRouter, Response
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from src.api.models.labeling import AnnotationSaveRequest, PointSegmentationRequest
from src.config import FAST_SAM_CHECKPOINT, BaseContext, Request, Template, templates
from src.core.utils import base64_to_numpy, is_hx_request
from src.db import CalibrationRecording, engine, SimRoom
from ultralytics import FastSAM

router = APIRouter(prefix="/labeling")


@dataclass
class LabelingContext(BaseContext):
    recording_uuid: str | None = None
    calibration_recording: CalibrationRecording | None = None
    annotations: list[dict[str, str | list[tuple[float, float]] | list[int]]] = field(default_factory=list)
    classes: list[dict] = field(default_factory=list)
    content: str = Template.LABELER

async def ensure_inference_running(request: Request) -> tuple[bool, str]:
    """Ensure inference model is loaded and running"""
    if not request.app.is_inference_running or request.app.inference_model is None:
        try:
            request.app.inference_model = FastSAM(FAST_SAM_CHECKPOINT)
            request.app.is_inference_running = True
            return True, ""
        except Exception as e:
            return False, f"Failed to load inference model: {e!s}"
    return True, ""


@router.get("/", response_class=HTMLResponse)
async def labeling(request: Request, calibration_id: int | None = None):
    context = LabelingContext(request=request)

    # First ensure inference is running
    is_running, error = await ensure_inference_running(request)
    if not is_running:
        return Response(status_code=503, content=f"Error: {error}")

    if not calibration_id:
        return Response(status_code=400, content="Error: No calibration_id provided")

    with Session(engine) as session:
        cal_rec = (
            session.query(CalibrationRecording)
            .filter(CalibrationRecording.id == calibration_id)
            .first()
        )

        if not cal_rec:
            return Response(status_code=404, content="Error: Calibration recording not found")

        sim_room = session.query(SimRoom).get(cal_rec.sim_room_id)
        if not sim_room:
            return Response(status_code=404, content="Error: Sim Room not found")

        context.annotations = [annotation.to_dict() for annotation in cal_rec.annotations]
        context.calibration_recording = cal_rec
        context.recording_uuid = cal_rec.recording_uuid
        context.classes = [cls.to_dict() for cls in sim_room.classes]

    if is_hx_request(request):
        return templates.TemplateResponse(Template.LABELER, context.to_dict())
    return templates.TemplateResponse(Template.INDEX, context.to_dict())


@router.post("/segmentation", response_class=JSONResponse)
async def get_point_segmentation(request: Request, body: PointSegmentationRequest):
    image = base64_to_numpy(body.frame)
    results = []

    # for annotation in body.annotations:
    #     # Use model helper methods to get and format segmentation results
    #     mask = fastsam.get_mask(
    #         image=image,
    #         model=request.app.inference_model,
    #         points=annotation.points,
    #         point_labels=annotation.point_labels,
    #         conf=0.6,
    #         iou=0.9
    #     )

    #     results.append(Annotation.create_segmentation_result(
    #         label=annotation.label,
    #         mask=mask,
    #         point_labels=annotation.point_labels
    #     ))

    return JSONResponse(content=results)


@router.post("/{calibration_id}/annotations", response_class=JSONResponse)
async def save_annotations(request: Request, calibration_id: int, body: AnnotationSaveRequest):
    cal_rec = CalibrationRecording.get_for_labeling(calibration_id)
    if not cal_rec:
        return Response(status_code=404, content="Calibration recording not found")

    try:
        cal_rec.save_annotations(body.frame_id, body.annotations)
        return JSONResponse(content={"success": True})
    except Exception as e:
        return Response(status_code=500, content=f"Failed to save annotations: {e!s}")
