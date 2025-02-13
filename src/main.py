from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

import src.logic.glasses as glasses
from src.api import calibration_recording, labeling, recordings, simrooms
from src.api.models import App, GlassesConnectionContext, Request
from src.config import Template, templates
from src.db.db import init_database
from src.db.models.recording import Recording


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    # Drop and recreate all database tables to ensure schema is up to date
    init_database()
    Recording.clean_recordings()
    yield


app = App(lifespan=lifespan)
app.include_router(recordings.router)
# app.include_router(labeling.router)
# app.include_router(simrooms.router)
# app.include_router(calibration_recording.router)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(Template.INDEX, {"request": request})


@app.get("/glasses/connection", response_class=HTMLResponse)
async def glasses_connection(request: Request):
    """Retrieve connection details for the glasses"""
    context = GlassesConnectionContext(
        request=request, glasses_connected=await glasses.is_connected(), battery_level=await glasses.get_battery_level()
    )
    return templates.TemplateResponse(Template.CONNECTION_STATUS, context.to_dict())
