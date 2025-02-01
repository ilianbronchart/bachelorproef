from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Optional

import src.logic.glasses as glasses
from fastapi import APIRouter, Form, Request, Response
from fastapi.responses import HTMLResponse
from src.api.settings import BaseContext, Template, app, templates
from src.logic.glasses.recording import recording_exists

router = APIRouter(prefix="/labeling")

@dataclass
class LabelingContext(BaseContext):
    local_recordings: Optional[list[dict[str, Any]]] = None
    recording: Optional[dict[str, Any]] = None
    error_msg: Optional[str] = None
    content: str = Template.LABELING


@router.get("/", response_class=HTMLResponse)
async def labeling(
    request: Request, 
    recording_uuid: Optional[Annotated[str, Form()]] = None
):
    context = LabelingContext(request=request)

    print("AMONG US?", app.is_inference_running)

    if recording_uuid:
        if not recording_exists(recording_uuid):
            context.error_msg = "Recording does not exist"    
        elif app.is_inference_running:
            context.recording = glasses.get_local_recording(recording_uuid)
        else:
            context.error_msg = "Inference is not running, please try again"

    if not recording_uuid or context.error_msg:
        recordings = [rec for rec in glasses.get_local_recordings()]
        recordings.sort(key=lambda x: datetime.fromisoformat(x.created), reverse=True)
        context.local_recordings = [rec.get_formatted() for rec in recordings]

    return templates.TemplateResponse(Template.INDEX, context.to_dict())
