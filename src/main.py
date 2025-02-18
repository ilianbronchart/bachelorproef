from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

import src.logic.glasses as glasses
from src.api import labeling, recordings, simrooms
from src.api.models import App, GlassesConnectionContext, Request
from src.config import Template, templates
from src.db.db import engine, Base
from src.db.models import Recording, Annotation, PointLabel

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    Base.metadata.drop_all(bind=engine, tables=[PointLabel.__table__])
    Base.metadata.drop_all(bind=engine, tables=[Annotation.__table__])
    Base.metadata.create_all(bind=engine)
    Recording.clean_recordings()
    yield


app = App(lifespan=lifespan)
app.include_router(recordings.router)
app.include_router(labeling.router)
app.include_router(simrooms.router)


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
