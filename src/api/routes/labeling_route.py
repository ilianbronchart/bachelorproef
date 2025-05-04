from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Response
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.models import (
    Request,
)
from src.api.repositories import annotations_repo
from src.api.services import labeler_service
from src.api.utils import image_utils
from src.config import Template, templates
from src.utils import is_hx_request

router = APIRouter(prefix="/labeling")


@router.post("/", response_class=Response)
async def start_labeling(
    request: Request, calibration_id: int, db: Session = Depends(get_db)
) -> Response:
    labeler_service.load(db, calibration_id)
    labeling_context = labeler_service.get_labeling_context(request).model_dump()

    if is_hx_request(request):
        return templates.TemplateResponse(Template.LABELER, labeling_context)
    return templates.TemplateResponse(Template.INDEX, labeling_context)


@router.get("/", response_class=HTMLResponse)
async def labeling(request: Request) -> HTMLResponse:
    labeling_context = labeler_service.get_labeling_context(request).model_dump()
    if is_hx_request(request):
        return templates.TemplateResponse(Template.LABELER, labeling_context)
    return templates.TemplateResponse(Template.INDEX, labeling_context)


@router.get("/point_labels", response_class=JSONResponse)
async def point_labels(db: Session = Depends(get_db)) -> JSONResponse:
    labeler_service.get_point_labels(db)
    point_labels = labeler_service.get_point_labels(db)
    return JSONResponse(content=[pl.model_dump() for pl in point_labels])


@router.get("/current_frame", response_class=Response)
async def current_frame() -> Response:
    frame = labeler_service.get_current_frame_overlay()
    bytes = image_utils.encode_to_png_bytes(frame)
    return Response(content=bytes, media_type="image/png")


@router.get("/timeline", response_class=HTMLResponse)
async def timeline(
    request: Request,
    polling: bool,
    frame_idx: int | None = None,
    db: Session = Depends(get_db),
) -> HTMLResponse:
    timeline_context = labeler_service.get_timeline_context(
        request, db, polling, frame_idx
    )
    return templates.TemplateResponse(
        Template.LABELING_TIMELINE, context=timeline_context.model_dump()
    )


@router.get("/classes", response_class=HTMLResponse)
async def classes(
    request: Request,
    selected_class_id: int = -1,
    db: Session = Depends(get_db),
) -> HTMLResponse:
    context = labeler_service.get_classes_context(request, db, selected_class_id)
    return templates.TemplateResponse(
        Template.LABELING_CLASSES, context=context.model_dump()
    )


@router.get("/annotations", response_class=HTMLResponse)
async def annotations(
    request: Request,
    db: Session = Depends(get_db),
) -> HTMLResponse:
    context = labeler_service.get_annotations_context(request, db)
    return templates.TemplateResponse(Template.LABELING_ANNOTATIONS, context.model_dump())


@dataclass
class AnnotationPostBody:
    point: tuple[int, int]
    label: int
    delete_point: bool = False


@router.post("/annotations", response_class=Response)
async def post_annotation(
    body: AnnotationPostBody,
    db: Session = Depends(get_db),
) -> Response:
    labeler_service.post_annotation(db, body.point, body.label, body.delete_point)
    return await current_frame()


@router.delete("/annotations/{annotation_id}", response_class=HTMLResponse)
async def delete_calibration_annotation(
    request: Request, annotation_id: int, db: Session = Depends(get_db)
) -> HTMLResponse:
    annotations_repo.delete_annotation(db, annotation_id)
    return await annotations(request, db)


@router.post("/tracking", response_class=HTMLResponse)
async def tracking(
    request: Request,
    db: Session = Depends(get_db),
) -> HTMLResponse:
    labeler_service.start_tracking(db)
    return await timeline(request, polling=False, db=db)


@router.post("/settings", response_class=HTMLResponse)
async def settings(
    request: Request,
    show_inactive_classes: Annotated[bool, Form()],
) -> HTMLResponse:
    labeler_service.post_settings(
        show_inactive_classes=show_inactive_classes,
    )
    context = labeler_service.get_labeling_context(request)
    return templates.TemplateResponse(
        Template.LABELING_SETTINGS, context=context.model_dump()
    )


@router.get("/settings", response_class=HTMLResponse)
async def settings(
    request: Request,
) -> HTMLResponse:
    context = labeler_service.get_labeling_context(request)
    return templates.TemplateResponse(
        Template.LABELING_SETTINGS, context=context.model_dump()
    )
