from dataclasses import dataclass
from typing import Annotated, cast

from fastapi import APIRouter, Depends, Form, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session

from src.api.db import get_db
from src.api.exceptions import (
    LabelingServiceNotAvailableError,
    NoClassSelectedError,
    TrackingJobAlreadyRunningError,
)
from src.api.models import App
from src.api.models.context import (
    LabelingAnnotationsContext,
    LabelingClassesContext,
    LabelingContext,
    LabelingTimelineContext,
)
from src.api.repositories import annotations_repo
from src.api.services import annotations_service, simrooms_service
from src.api.services.labeling_service import Labeler
from src.api.utils import image_utils
from src.config import Template, templates
from src.utils import is_hx_request

router = APIRouter(prefix="/labeling")


def get_labeling_context(request: Request, labeler: Labeler) -> LabelingContext:
    return LabelingContext(
        request=request,
        simroom_id=labeler.simroom_id,
        recording_id=labeler.recording_id,
        show_inactive_classes=labeler.show_inactive_classes,
    )


def require_labeler(request: Request) -> Labeler:
    app = cast(App, request.app)  # Now MyPy knows what this is
    if app.labeler is None:
        raise LabelingServiceNotAvailableError()
    return app.labeler


@router.post("/", response_class=Response)
async def start_labeling(
    request: Request,
    calibration_id: int,
    db: Session = Depends(get_db),
) -> Response:
    cal_rec = simrooms_service.get_calibration_recording(
        db=db,
        calibration_id=calibration_id,
    )
    labeler = Labeler(cal_rec=cal_rec)
    request.app.labeler = labeler

    labeling_context = get_labeling_context(request, labeler).model_dump()
    if is_hx_request(request):
        return templates.TemplateResponse(Template.LABELER, labeling_context)
    return templates.TemplateResponse(Template.INDEX, labeling_context)


@router.get("/", response_class=HTMLResponse)
async def labeling(
    request: Request, labeler: Labeler = Depends(require_labeler)
) -> HTMLResponse:
    labeling_context = get_labeling_context(request, labeler).model_dump()
    if is_hx_request(request):
        return templates.TemplateResponse(Template.LABELER, labeling_context)
    return templates.TemplateResponse(Template.INDEX, labeling_context)


@router.get("/point_labels", response_class=JSONResponse)
async def point_labels(
    db: Session = Depends(get_db), labeler: Labeler = Depends(require_labeler)
) -> JSONResponse:
    point_labels = annotations_service.get_point_labels(
        db=db,
        calibration_id=labeler.calibration_id,
        frame_idx=labeler.current_frame_idx,
    )
    return JSONResponse(content=[pl.model_dump() for pl in point_labels])


@router.get("/current_frame", response_class=Response)
async def current_frame(
    db: Session = Depends(get_db), labeler: Labeler = Depends(require_labeler)
) -> Response:
    frame = labeler.get_current_frame_overlay(db=db)
    png_bytes = image_utils.encode_to_png_bytes(frame)
    return Response(content=png_bytes, media_type="image/png")


@router.get("/timeline", response_class=HTMLResponse)
async def timeline(
    request: Request,
    polling: bool,
    frame_idx: int | None = None,
    db: Session = Depends(get_db),
    labeler: Labeler = Depends(require_labeler),
) -> HTMLResponse:
    frame_idx = labeler.current_frame_idx if frame_idx is None else frame_idx
    selected_class_id = labeler.selected_class_id

    context = LabelingTimelineContext(
        request=request,
        current_frame_idx=frame_idx,
        frame_count=labeler.frame_count,
        selected_class_id=selected_class_id,
    )

    if labeler.has_selected_class:
        simroom_class = simrooms_service.get_simroom_class(
            db=db, class_id=selected_class_id
        )
        context.tracks = annotations_repo.get_tracks(labeler.current_class_results_path)
        context.selected_class_color = simroom_class.color

    if labeler.is_tracking_current_class and labeler.tracking_progress is not None:
        context.tracking_progress = labeler.tracking_progress
        context.is_tracking = True

    if not polling:
        labeler.seek(frame_idx)
        context.update_canvas = True

    return templates.TemplateResponse(
        Template.LABELING_TIMELINE, context=context.model_dump()
    )


@router.get("/classes", response_class=HTMLResponse)
async def classes(
    request: Request,
    selected_class_id: int | None = None,
    db: Session = Depends(get_db),
    labeler: Labeler = Depends(require_labeler),
) -> HTMLResponse:
    classes = simrooms_service.get_simroom_classes(db=db, simroom_id=labeler.simroom_id)

    if selected_class_id is None:
        selected_class_id = classes[0].id if classes else None

    labeler.set_selected_class_id(db, selected_class_id)

    context = LabelingClassesContext(
        request=request,
        selected_class_id=labeler.selected_class_id,
        simroom_id=labeler.simroom_id,
        classes=classes,
    )

    return templates.TemplateResponse(
        Template.LABELING_CLASSES, context=context.model_dump()
    )


@router.get("/annotations", response_class=HTMLResponse)
async def annotations(
    request: Request,
    db: Session = Depends(get_db),
    labeler: Labeler = Depends(require_labeler),
) -> HTMLResponse:
    annotations = annotations_service.get_annotations_by_class_id(
        db=db,
        calibration_id=labeler.calibration_id,
        class_id=labeler.selected_class_id,
    )
    context = LabelingAnnotationsContext(
        request=request,
        annotations=annotations,
    )

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
    labeler: Labeler = Depends(require_labeler),
) -> Response:
    if not labeler.has_selected_class:
        raise NoClassSelectedError()

    annotations_service.post_annotation_point(
        db=db,
        frame=labeler.current_frame,
        image_predictor=labeler.image_predictor,
        calibration_id=labeler.calibration_id,
        frame_idx=labeler.current_frame_idx,
        class_id=labeler.selected_class_id,
        new_point=body.point,
        new_label=body.label,
        delete_point=body.delete_point,
    )

    return await current_frame(db, labeler)


@router.delete("/annotations/{annotation_id}", response_class=HTMLResponse)
async def delete_calibration_annotation(
    request: Request,
    annotation_id: int,
    db: Session = Depends(get_db),
    labeler: Labeler = Depends(require_labeler),
) -> HTMLResponse:
    annotations_repo.delete_annotation(db, annotation_id)
    return await annotations(request, db, labeler)


@router.post("/tracking", response_class=HTMLResponse)
async def tracking(
    request: Request,
    db: Session = Depends(get_db),
    labeler: Labeler = Depends(require_labeler),
) -> HTMLResponse:
    if labeler.is_tracking:
        raise TrackingJobAlreadyRunningError()

    if not labeler.has_selected_class:
        raise NoClassSelectedError()

    annotations = annotations_service.get_annotations_by_class_id(
        db=db,
        calibration_id=labeler.calibration_id,
        class_id=labeler.selected_class_id,
    )
    labeler.start_tracking(annotations)

    return await timeline(request, polling=False, db=db, labeler=labeler)


@router.post("/settings", response_class=HTMLResponse)
async def post_settings(
    request: Request,
    show_inactive_classes: Annotated[bool, Form()],
    labeler: Labeler = Depends(require_labeler),
) -> HTMLResponse:
    labeler.set_show_inactive_classes(show_inactive_classes)
    context = get_labeling_context(request, labeler)
    return templates.TemplateResponse(
        Template.LABELING_SETTINGS, context=context.model_dump()
    )


@router.get("/settings", response_class=HTMLResponse)
async def get_settings(
    request: Request,
    labeler: Labeler = Depends(require_labeler),
) -> HTMLResponse:
    context = get_labeling_context(request, labeler)
    return templates.TemplateResponse(
        Template.LABELING_SETTINGS, context=context.model_dump()
    )
