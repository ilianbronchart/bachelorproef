import src.logic.glasses as glasses
from fastapi import APIRouter, Response
from fastapi.responses import HTMLResponse
from src.api.models import RecordingsContext, Request
from src.config import Template, templates
from src.db import Recording
from src.utils import is_hx_request

router = APIRouter(prefix="/recordings")


@router.get("/")
async def recordings(request: Request) -> HTMLResponse:
    context = RecordingsContext(request=request)
    if is_hx_request(request):
        return templates.TemplateResponse(Template.RECORDINGS, context.to_dict())
    return templates.TemplateResponse(Template.INDEX, context.to_dict())


@router.get("/local")
async def local_recordings(request: Request) -> HTMLResponse:
    """Retrieve metadata for all recordings in the local directory"""
    context = RecordingsContext(request=request, recordings=Recording.get_all())

    if is_hx_request(request):
        return templates.TemplateResponse(Template.LOCAL_RECORDINGS, context.to_dict())
    return templates.TemplateResponse(Template.INDEX, context.to_dict())


@router.delete("/local/{recording_id}", response_class=HTMLResponse)
async def delete_local_recording(request: Request, recording_id: str):
    """Delete a recording from the local directory"""
    context = RecordingsContext(request=request)

    try:
        recording = Recording.get(recording_id)

        if recording is None:
            return Response(status_code=404, content="Error: Recording not found")

        recording.remove()
        context.recordings = Recording.get_all()
        return templates.TemplateResponse(Template.LOCAL_RECORDINGS, context.to_dict())
    except Exception:
        return Response(status_code=500, content="Error: Something went wrong, please try again later")


@router.get("/glasses", response_class=HTMLResponse)
async def glasses_recordings(request: Request):
    """Retrieve metadata for all recordings on the glasses"""
    context = RecordingsContext(request=request, glasses_connected=await glasses.is_connected())

    if not context.glasses_connected:
        context.failed_connection = True
        return templates.TemplateResponse(Template.GLASSES_RECORDINGS, context.to_dict(), status_code=503)

    context.recordings = await glasses.get_recordings()

    if is_hx_request(request):
        return templates.TemplateResponse(Template.GLASSES_RECORDINGS, context.to_dict())
    return templates.TemplateResponse(Template.INDEX, context.to_dict())


@router.get("/glasses/{recording_uuid}/download", response_class=HTMLResponse)
async def download_recording(request: Request, recording_uuid: str):
    """Download a recording from the glasses"""
    context = RecordingsContext(request=request, glasses_connected=await glasses.is_connected())

    if not context.glasses_connected:
        context.failed_connection = True
        return templates.TemplateResponse(Template.GLASSES_RECORDINGS, context.to_dict(), status_code=503)

    try:
        recording = await glasses.get_recording(recording_uuid)
        if recording.is_complete():
            return Response(status_code=409, content="Error: Recording already exists in local directory")
        await recording.download()

        context.recordings = Recording.get_all()
        return templates.TemplateResponse(Template.LOCAL_RECORDINGS, context.to_dict())
    except KeyError:
        return Response(status_code=404, content="Error: Recording not found on the glasses")
    except RuntimeError:
        return Response(status_code=500, content="Error: Failed to download recording")
    except Exception:
        return Response(status_code=500, content="Error: Something went wrong, please try again later")
