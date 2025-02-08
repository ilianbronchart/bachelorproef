from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

import src.logic.glasses as glasses
from src.api import labeling, recordings, simrooms
from src.config import App, BaseContext, Template, templates
from src.db.db import Base, engine


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    # Create all database tables if they do not exist yet
    Base.metadata.create_all(bind=engine)
    yield


app = App(lifespan=lifespan)
app.include_router(recordings.router)
app.include_router(labeling.router)
app.include_router(simrooms.router)



@dataclass
class GlassesConnectionContext(BaseContext):
    glasses_connected: bool
    battery_level: int


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
