from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy.sql import text

from src.api.db import Base, engine
from src.api.models import App, Request
from src.api.models.context import GlassesConnectionContext
from src.api.routes import labeling_route, recordings_route, simrooms_route
from src.api.services import glasses_service, recordings_service, simrooms_service
from src.api.services.labeling_service import Labeler
from src.config import Template, templates


@asynccontextmanager
async def lifespan(_app: App) -> AsyncGenerator[None, None]:
    with Session(engine) as session:
        recordings_service.clean_recordings(session)

    Base.metadata.create_all(bind=engine)

    with Session(engine) as db:
        cal_rec = simrooms_service.get_calibration_recording(
            db=db,
            calibration_id=1,
        )
        labeler = Labeler(cal_rec=cal_rec)
        _app.labeler = labeler

    yield


app = App(lifespan=lifespan)  # type: ignore[no-untyped-call]
app.include_router(recordings_route.router)
app.include_router(simrooms_route.router)
app.include_router(labeling_route.router)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(Template.INDEX, {"request": request})


@app.get("/glasses/connection", response_class=HTMLResponse)
async def glasses_connection(request: Request) -> HTMLResponse:
    """Retrieve connection details for the glasses"""
    context = GlassesConnectionContext(
        request=request,
        glasses_connected=await glasses_service.is_connected(),
        battery_level=await glasses_service.get_battery_level(),
    )
    return templates.TemplateResponse(Template.CONNECTION_STATUS, context.model_dump())
