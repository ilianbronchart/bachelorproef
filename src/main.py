from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from src.api.db import Base, engine
from src.api.models import App, Request
from src.api.models.context import GlassesConnectionContext
from src.api.routes import recordings_route, simrooms_route
from src.api.services import glasses_service, recordings_service
from src.config import Template, templates


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    with Session(engine) as session:
        recordings_service.clean_recordings(session)

    Base.metadata.create_all(bind=engine)
    yield


app = App(lifespan=lifespan)  # type: ignore[no-untyped-call]
app.include_router(recordings_route.router)
app.include_router(simrooms_route.router)
# app.include_router(labeling.router)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(Template.INDEX, {"request": request})


@app.get("/glasses/connection", response_class=HTMLResponse)
async def glasses_connection(request: Request) -> HTMLResponse:
    """Retrieve connection details for the glasses"""
    context = GlassesConnectionContext(
        _request=request,
        glasses_connected=await glasses_service.is_connected(),
        battery_level=await glasses_service.get_battery_level(),
    )
    return templates.TemplateResponse(Template.CONNECTION_STATUS, context.model_dump())
