from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any

import src.logic.glasses as glasses
from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse
from src.config import BaseContext, Request, Template, templates
from src.core.utils import is_hx_request
from src.db import Recording

router = APIRouter(prefix="/labeling")


@dataclass
class LabelingContext(BaseContext):
    local_recordings: list[dict[str, Any]] | None = None
    recording: dict[str, Any] | None = None
    error_msg: str | None = None
    content: str = Template.LABELING


@router.get("/", response_class=HTMLResponse)
async def labeling(request: Request, recording_uuid: Annotated[str, Form()] | None = None):
    context = LabelingContext(request=request)

    if recording_uuid:
        recording = Recording.get(recording_uuid)
        if recording is None:
            context.error_msg = "Recording does not exist"
        elif request.app.is_inference_running:
            context.recording = glasses.get_local_recording(recording_uuid)
        else:
            context.error_msg = "Inference is not running, please try again"

    if not recording_uuid or context.error_msg:
        recordings = [rec for rec in glasses.get_local_recordings()]
        recordings.sort(key=lambda x: datetime.fromisoformat(x.created), reverse=True)
        context.local_recordings = [rec.get_formatted() for rec in recordings]

    if is_hx_request(request):
        return templates.TemplateResponse(Template.LABELING, context.to_dict())
    return templates.TemplateResponse(Template.INDEX, context.to_dict())
