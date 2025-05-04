from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from src.api.db import get_db
from src.api.models.context import RecordingsContext
from src.api.repositories import recordings_repo
from src.api.services import glasses_service, recordings_service
from src.config import Template, templates
from src.utils import is_hx_request

router = APIRouter(prefix="/recordings")


@router.get("/")
async def recordings(request: Request) -> HTMLResponse:
    context = RecordingsContext(request=request)
    if is_hx_request(request):
        return templates.TemplateResponse(Template.RECORDINGS, context.model_dump())
    return templates.TemplateResponse(Template.INDEX, context.model_dump())


@router.get("/local")
async def local_recordings(
    request: Request, db: Session = Depends(get_db)
) -> HTMLResponse:
    """Retrieve metadata for all recordings in the local directory"""
    recordings = recordings_service.get_all(db)
    context = RecordingsContext(request=request, recordings=recordings)

    if is_hx_request(request):
        return templates.TemplateResponse(Template.LOCAL_RECORDINGS, context.model_dump())
    return templates.TemplateResponse(Template.INDEX, context.model_dump())


@router.delete("/local/{recording_id}", response_class=HTMLResponse)
async def delete_local_recording(
    request: Request, recording_id: str, db: Session = Depends(get_db)
) -> HTMLResponse:
    """Delete a recording from the local directory"""
    recordings_repo.delete(db, recording_id)
    recordings = recordings_service.get_all(db)
    context = RecordingsContext(request=request, recordings=recordings)
    return templates.TemplateResponse(Template.LOCAL_RECORDINGS, context.model_dump())


@router.get("/glasses", response_class=HTMLResponse)
async def glasses_recordings(request: Request) -> HTMLResponse:
    """Retrieve metadata for all recordings on the glasses"""
    glasses_connected = await glasses_service.is_connected()
    context = RecordingsContext(request=request, glasses_connected=glasses_connected)

    if not context.glasses_connected:
        context.failed_connection = True
        return templates.TemplateResponse(
            Template.GLASSES_RECORDINGS, context.model_dump(), status_code=503
        )

    context.recordings = await glasses_service.get_recordings()

    if is_hx_request(request):
        return templates.TemplateResponse(
            Template.GLASSES_RECORDINGS, context.model_dump()
        )
    return templates.TemplateResponse(Template.INDEX, context.model_dump())


@router.get("/glasses/{recording_id}/download", response_class=HTMLResponse)
async def download_recording(
    request: Request, recording_id: str, db: Session = Depends(get_db)
) -> HTMLResponse:
    """Download a recording from the glasses"""
    glasses_connected = await glasses_service.is_connected()
    context = RecordingsContext(request=request, glasses_connected=glasses_connected)

    if not context.glasses_connected:
        context.failed_connection = True
        return templates.TemplateResponse(
            Template.GLASSES_RECORDINGS, context.model_dump(), status_code=503
        )

    await glasses_service.download_recording(db, recording_id)
    context.recordings = recordings_service.get_all(db)
    return templates.TemplateResponse(Template.LOCAL_RECORDINGS, context.model_dump())
