from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

import src.logic.glasses as glasses
from src.api import labeling, recordings, simrooms
from src.api.models import App, GlassesConnectionContext, Request
from src.api.models.labeler import Labeler
from src.config import Template, templates
from src.db.db import Base, engine
from src.db.models import Recording
from src.db.models.calibration import CalibrationRecording


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    # Base.metadata.drop_all(bind=engine, tables=[PointLabel.__table__])
    # Base.metadata.drop_all(bind=engine, tables=[Annotation.__table__])
    Base.metadata.create_all(bind=engine)
    Recording.clean_recordings()
    yield


app = App(lifespan=lifespan)  # type: ignore[no-untyped-call]
app.include_router(recordings.router)
app.include_router(labeling.router)
app.include_router(simrooms.router)

with Session(engine) as session:
    cal_rec = session.query(CalibrationRecording).first()
    app.labeler = Labeler(calibration_recording=cal_rec)  # type: ignore[arg-type]


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(Template.INDEX, {"request": request})


@app.get("/glasses/connection", response_class=HTMLResponse)
async def glasses_connection(request: Request) -> HTMLResponse:
    """Retrieve connection details for the glasses"""
    context = GlassesConnectionContext(
        request=request, glasses_connected=await glasses.is_connected(), battery_level=await glasses.get_battery_level()
    )
    return templates.TemplateResponse(Template.CONNECTION_STATUS, context.to_dict())
