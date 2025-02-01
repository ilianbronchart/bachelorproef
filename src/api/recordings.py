import src.logic.glasses as glasses
from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse
from src.api.settings import Template, templates
from src.core import DotDict as dd
from src.core.utils import is_hx_request

router = APIRouter(prefix="/recordings")


@router.get("/", response_class=HTMLResponse)
async def recordings(request: Request):
    context = dd()
    context.request = request
    context.local_recordings = [rec.get_formatted() for rec in glasses.get_local_recordings()]

    if is_hx_request(request):
        return templates.TemplateResponse(Template.RECORDINGS, context)

    context.content = Template.RECORDINGS
    return templates.TemplateResponse(Template.INDEX, context)


@router.get("/local", response_class=HTMLResponse)
async def local_recordings(request: Request):
    """Retrieve metadata for all recordings in the local directory"""
    context = dd()
    context.request = request
    context.local_recordings = glasses.get_local_recordings()

    if is_hx_request(request):
        return templates.TemplateResponse(Template.LOCAL_RECORDINGS, context)

    # TODO if not hx request, return 404 page


@router.delete("/local/{recording_id}", response_class=HTMLResponse)
async def delete_local_recording(request: Request, response: Response, recording_id: str):
    """Delete a recording from the local directory"""
    context = dd()
    context.request = request
    context.local_recordings = glasses.get_local_recordings()

    try:
        glasses.delete_local_recording(recording_id)
        context.local_recordings = glasses.get_local_recordings()
        return templates.TemplateResponse(Template.LOCAL_RECORDINGS, context)
    except Exception:
        return Response(status_code=500, content="Error: Something went wrong, please try again later")


@router.get("/local", response_class=HTMLResponse)
async def local_recordings(request: Request):
    """Retrieve metadata for all recordings in the local directory"""
    context = dd()
    context.request = request
    context.local_recordings = glasses.get_local_recordings()

    if is_hx_request(request):
        return templates.TemplateResponse(Template.LOCAL_RECORDINGS, context)

    # TODO if not hx request, return 404 page


@router.delete("/local/{recording_id}", response_class=HTMLResponse)
async def delete_local_recording(request: Request, recording_id: str):
    """Delete a recording from the local directory"""
    context = dd()
    context.request = request
    context.local_recordings = glasses.get_local_recordings()

    try:
        glasses.delete_local_recording(recording_id)
        context.local_recordings = glasses.get_local_recordings()
        return templates.TemplateResponse(Template.LOCAL_RECORDINGS, context)
    except Exception:
        return Response(status_code=500, content="Error: Something went wrong, please try again later")


@router.get("/glasses", response_class=HTMLResponse)
async def glasses_recordings(request: Request, response: Response):
    """Retrieve metadata for all recordings on the glasses"""
    context = dd()
    context.request = request
    context.glasses_connected = await glasses.is_connected()
    context.table_type = "glasses"

    if context.glasses_connected:
        context.glasses_recordings = [rec.get_formatted() for rec in await glasses.get_glasses_recordings()]
    else:
        context.target_url = "/recordings/glasses"
        context.error_msg = "Failed to connect to Tobii Glasses"
        context.retry_target = "#glasses-recordings"
        return templates.TemplateResponse(
            Template.FAILED_CONNECTION, context, headers=response.headers, status_code=503
        )

    if is_hx_request(request):
        return templates.TemplateResponse(Template.GLASSES_RECORDINGS, context)

    # TODO if not hx request, return 404 page


@router.get("/glasses/{recording_id}/download", response_class=HTMLResponse)
async def download_recording(request: Request, recording_id: str):
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
        recording = await glasses.get_recording(recording_id)
        await glasses.download_recording(recording)
        context.local_recordings = [rec.get_formatted() for rec in glasses.get_local_recordings()]
        return templates.TemplateResponse(Template.LOCAL_RECORDINGS, context, status_code=200)
    except KeyError:
        return Response(status_code=404, content="Error: Recording not found on the glasses")
    except RuntimeError:
        return Response(status_code=500, content="Error: Failed to download recording")
    except ValueError:
        return Response(status_code=409, content="Error: Recording already exists in local directory")
    except Exception:
        return Response(status_code=500, content="Error: Something went wrong, please try again later")
