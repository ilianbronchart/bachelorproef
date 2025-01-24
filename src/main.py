from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import src.logic.glasses as glasses
from src.core import DotDict as dd
from src.core.utils import is_hx_request

app = FastAPI()
app.mount("/static", StaticFiles(directory="src/static", html=True), name="static")

templates = Jinja2Templates(directory="src/templates")

glasses.clean_local_recordings()


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.jinja", {"request": request})


@app.get("/recording", response_class=HTMLResponse)
async def recordings(request: Request):
    context = dd()
    context.request = request
    context.local_recordings = [rec.get_formatted() for rec in await glasses.get_local_recordings()]

    if is_hx_request(request):
        return templates.TemplateResponse("pages/recordings.jinja", context)

    context.content = "pages/recordings.jinja"
    return templates.TemplateResponse("index.jinja", context)


@app.get("/local/recordings", response_class=HTMLResponse)
async def local_recordings(request: Request):
    """Retrieve metadata for all recordings in the local directory"""
    context = dd()
    context.request = request
    context.local_recordings = await glasses.get_local_recordings()

    if is_hx_request(request):
        return templates.TemplateResponse("components/local-recordings.jinja", context)

    # TODO if not hx request, return 404 page


@app.delete("/local/recordings/{recording_id}", response_class=HTMLResponse)
async def delete_local_recording(request: Request, recording_id: str):
    """Delete a recording from the local directory"""
    context = dd()
    context.request = request
    context.local_recordings = await glasses.get_local_recordings()

    try:
        glasses.delete_local_recording(recording_id)
        context.local_recordings = await glasses.get_local_recordings()
        return templates.TemplateResponse("components/local-recordings.jinja", context)
    except Exception:
        return Response(status_code=500, content="Error: Something went wrong, please try again later")


@app.get("/glasses/recordings", response_class=HTMLResponse)
async def glasses_recordings(request: Request, response: Response):
    """Retrieve metadata for all recordings on the glasses"""
    context = dd()
    context.request = request
    context.glasses_connected = await glasses.is_connected()
    context.table_type = "glasses"

    if context.glasses_connected:
        context.glasses_recordings = [rec.get_formatted() for rec in await glasses.get_glasses_recordings()]
    else:
        context.target_url = "/glasses/recordings"
        context.error_message = "Failed to connect to Tobii Glasses"
        context.retry_target = "#glasses-recordings"
        return templates.TemplateResponse(
            "components/failed-connection.jinja", context, headers=response.headers, status_code=503
        )

    if is_hx_request(request):
        return templates.TemplateResponse("components/glasses-recordings.jinja", context)

    # TODO if not hx request, return 404 page


@app.get("/glasses/recordings/{recording_id}/download", response_class=HTMLResponse)
async def download_recording(request: Request, recording_id: str):
    """Download a recording from the glasses"""
    context = dd()
    context.request = request
    context.glasses_connected = await glasses.is_connected()

    if not context.glasses_connected:
        context.target_url = "/glasses/recordings"
        context.error_message = "Could not download recording. Tobii Glasses are not connected."
        context.retry_target = "#glasses-recordings"
        return templates.TemplateResponse("components/failed-connection.jinja", context, status_code=503)
    try:
        recording = await glasses.get_recording(recording_id)
        await glasses.download_recording(recording)
        context.local_recordings = [rec.get_formatted() for rec in await glasses.get_local_recordings()]
        return templates.TemplateResponse("components/local-recordings.jinja", context, status_code=200)
    except KeyError:
        return Response(status_code=404, content="Error: Recording not found on the glasses")
    except RuntimeError:
        return Response(status_code=500, content="Error: Failed to download recording")
    except ValueError:
        return Response(status_code=409, content="Error: Recording already exists in local directory")
    except Exception:
        return Response(status_code=500, content="Error: Something went wrong, please try again later")


@app.get("/glasses/connection", response_class=HTMLResponse)
async def glasses_connection(request: Request):
    """Retrieve connection details for the glasses"""
    context = dd()
    context.request = request
    context.glasses_connected = await glasses.is_connected()
    context.battery_level = await glasses.get_battery_level()
    return templates.TemplateResponse("components/connection-status.jinja", context)
