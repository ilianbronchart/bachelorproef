
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.core import DotDict as dd
from src.core.utils import is_hx_request
from src.logic.glasses import get_glasses_recordings, get_local_recordings, is_glasses_connected

app = FastAPI()
app.mount("/static", StaticFiles(directory="src/static", html=True), name="static")

templates = Jinja2Templates(directory="src/templates")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.jinja", {"request": request})


@app.get("/recordings", response_class=HTMLResponse)
async def recordings(request: Request):
    context = dd()
    context.request = request
    context.local_recordings = await get_local_recordings()

    if is_hx_request(request):
        return templates.TemplateResponse("pages/recordings.jinja", context)

    context.content = "pages/recordings.jinja"
    return templates.TemplateResponse("index.jinja", context)


@app.get("/glasses/recordings", response_class=HTMLResponse)
async def glasses_recordings(request: Request):
    """Retrieve metadata for all recordings on the glasses"""
    context = dd()
    context.request = request
    context.glasses_connected = await is_glasses_connected()

    if context.glasses_connected:
        context.glasses_recordings = [rec.get_formatted() for rec in await get_glasses_recordings()]

    if is_hx_request(request):
        return templates.TemplateResponse("components/glasses-recordings.jinja", context)

    # TODO if not hx request, return 404 page
