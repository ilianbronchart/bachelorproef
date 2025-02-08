import src.logic.glasses as glasses
from fastapi import APIRouter, Response
from fastapi.responses import HTMLResponse
from src.config import Request, Template, templates
from src.core import DotDict as dd
from src.core.utils import is_hx_request
from src.db import Recording

router = APIRouter(prefix="/recordings")


@router.get("/", response_class=HTMLResponse)
async def recordings(request: Request):
    context = dd()
    context.request = request
    context.local_recordings = [rec.to_dict() for rec in Recording.get_all()]

    if is_hx_request(request):
        return templates.TemplateResponse(Template.RECORDINGS, context)

    context.content = Template.RECORDINGS
    return templates.TemplateResponse(Template.INDEX, context)


@router.get("/local", response_class=HTMLResponse)
async def local_recordings(request: Request):
    """Retrieve metadata for all recordings in the local directory"""
    context = dd()
    context.request = request
    context.local_recordings = [rec.to_dict() for rec in Recording.get_all()]

    if is_hx_request(request):
        return templates.TemplateResponse(Template.LOCAL_RECORDINGS, context)

    # TODO if not hx request, return 404 page


@router.delete("/local/{recording_id}", response_class=HTMLResponse)
async def delete_local_recording(request: Request, recording_id: str):
    """Delete a recording from the local directory"""
    context = dd()
    context.request = request

    try:
        recording = Recording.get(recording_id)
        if recording is None:
            return Response(status_code=404, content="Error: Recording not found")

        recording.remove()
        context.local_recordings = [rec.to_dict() for rec in Recording.get_all()]
        return templates.TemplateResponse(Template.LOCAL_RECORDINGS, context)
    except Exception as e:
        print(e)
        return Response(status_code=500, content="Error: Something went wrong, please try again later")


@router.get("/glasses", response_class=HTMLResponse)
async def glasses_recordings(request: Request, response: Response):
    """Retrieve metadata for all recordings on the glasses"""
    context = dd()
    context.request = request
    context.glasses_connected = await glasses.is_connected()

    if context.glasses_connected:
        context.glasses_recordings = [rec.to_dict() for rec in await glasses.get_recordings()]
    else:
        context.target_url = "/recordings/glasses"
        context.error_msg = "Could not connect to Tobii Glasses"
        context.retry_target = "#glasses-recordings"
        return templates.TemplateResponse(
            Template.FAILED_CONNECTION, context, headers=response.headers, status_code=503
        )

    if is_hx_request(request):
        return templates.TemplateResponse(Template.GLASSES_RECORDINGS, context)

    # TODO if not hx request, return 404 page


@router.get("/glasses/{recording_uuid}/download", response_class=HTMLResponse)
async def download_recording(request: Request, recording_uuid: str):
    """Download a recording from the glasses"""
    context = dd()
    context.request = request
    context.glasses_connected = await glasses.is_connected()

    if not context.glasses_connected:
        context.target_url = "/recordings/glasses"
        context.error_msg = "Could not download recording. Tobii Glasses are not connected."
        context.retry_target = "#glasses-recordings"
        return templates.TemplateResponse(Template.FAILED_CONNECTION, context, status_code=503)
    try:
        recording = await glasses.get_recording(recording_uuid)
        if recording.is_complete():
            return Response(status_code=409, content="Error: Recording already exists in local directory")

        await recording.download()

        context.local_recordings = [rec.to_dict() for rec in Recording.get_all()]
        return templates.TemplateResponse(Template.LOCAL_RECORDINGS, context, status_code=200)
    except KeyError:
        return Response(status_code=404, content="Error: Recording not found on the glasses")
    except RuntimeError:
        return Response(status_code=500, content="Error: Failed to download recording")
    except Exception:
        return Response(status_code=500, content="Error: Something went wrong, please try again later")
