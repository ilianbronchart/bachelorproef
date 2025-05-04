import base64
import json
import math
import tempfile
import threading
from pathlib import Path

import numpy as np
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sqlalchemy.orm import Session

from src.aliases import UInt8Array
from src.api.exceptions import (
    LabelingServiceNotAvailableError,
    NoClassSelectedError,
    NotFoundError,
    TrackingJobAlreadyRunningError,
)
from src.api.jobs.labeler_tracking import TrackingJob
from src.api.models.context import (
    LabelingAnnotation,
    LabelingAnnotationsContext,
    LabelingClassesContext,
    LabelingContext,
    LabelingTimelineContext,
    Request,
)
from src.api.models.db import Annotation, CalibrationRecording, PointLabel
from src.api.models.pydantic import AnnotationDTO, PointLabelDTO
from src.api.repositories import annotations_repo, simrooms_repo
from src.api.services import sam2_service
from src.api.utils import image_utils
from src.config import (
    Sam2Checkpoints,
)
from src.utils import extract_frames_to_dir, get_frame_from_dir


class Labeler:
    tracking_job: TrackingJob | None = None
    selected_class_id: int = -1
    show_inactive_classes: bool = True

    def __init__(self, cal_rec: CalibrationRecording):
        self.cal_rec: CalibrationRecording = cal_rec
        self.frames_path: Path = Path(tempfile.mkdtemp())

        extract_frames_to_dir(
            video_path=self.cal_rec.video_path, frames_path=self.frames_path
        )

        self.frame_count = len(list(self.frames_path.glob("*.jpg")))
        self.current_frame_idx = 0
        self.image_predictor: SAM2ImagePredictor = sam2_service.load_predictor(
            Sam2Checkpoints.LARGE
        )

    @property
    def current_class_results_path(self) -> Path:
        return self.cal_rec.labeling_results_path / str(self.selected_class_id)

    def seek(self, frame_idx: int) -> None:
        if self.current_frame_idx == frame_idx:
            return

        self.current_frame_idx = frame_idx
        frame = get_frame_from_dir(frame_idx, self.frames_path)
        self.image_predictor.set_image(frame)


_labeler = None


def require_labeler() -> Labeler:
    global _labeler
    if _labeler is None:
        raise LabelingServiceNotAvailableError()
    return _labeler


def load(
    db: Session,
    calibration_id: int,
) -> Labeler:
    global labeler
    cal_rec = simrooms_repo.get_calibration_recording(
        db=db,
        calibration_id=calibration_id,
    )
    if labeler is None:
        labeler = Labeler(cal_rec)


def unload() -> None:
    global labeler
    pass  # TODO: Implement unloadlabeler


def get_labeling_context(request: Request) -> LabelingContext:
    labeler = require_labeler()

    return LabelingContext(
        request=request,
        sim_room_id=labeler.cal_rec.sim_room_id,
        recording_id=labeler.cal_rec.recording_id,
        show_inactive_classes=labeler.show_inactive_classes,
    )


class PointLabelWithClassID(PointLabelDTO):
    class_id: int


def get_point_labels(db: Session) -> list[PointLabelWithClassID]:
    labeler = require_labeler()

    annotations = annotations_repo.get_annotations_by_frame_idx(
        db=db,
        calibration_recording_id=labeler.cal_rec.id,
        frame_idx=labeler.current_frame_idx,
    )

    point_labels = []
    for annotation in annotations:
        for point_label in annotation.point_labels:
            dto = PointLabelDTO.from_orm(point_label)  # or from_orm if your config allows
            with_class = PointLabelWithClassID.model_validate(
                dto.model_dump() | {"class_id": annotation.sim_room_class_id}
            )
            point_labels.append(with_class)

    return point_labels


def get_overlay_draw_data(db: Session) -> list[tuple[str, str, UInt8Array, UInt8Array]]:
    # Get all the annotations for the current frame
    current_frame_annotations = annotations_repo.get_annotations_by_frame_idx(
        db=db,
        calibration_recording_id=labeler.cal_rec.id,
        frame_idx=labeler.current_frame_idx,
    )
    annotation_class_ids = {ann.sim_room_class_id for ann in current_frame_annotations}

    # Get all tracked classes and filter out the ones which have an annotation in the current frame
    tracked_classes = simrooms_repo.get_tracked_classes(
        db=db,
        calibration_id=labeler.cal_rec.id,
    )
    tracked_classes = [
        cls_ for cls_ in tracked_classes if cls_.id not in annotation_class_ids
    ]

    # Filter out annotations and classes that are not active if show_inactive_classes is False
    if not labeler.show_inactive_classes:
        active_id = labeler.selected_class_id
        current_frame_annotations = [
            ann for ann in current_frame_annotations if ann.sim_room_class_id != active_id
        ]
        tracked_classes = [cls_ for cls_ in tracked_classes if cls_.id != active_id]

    # Information needed to draw the overlay (class name, color, mask, box)
    results_to_draw: list[tuple[str, str, UInt8Array, UInt8Array]] = []

    # Add current frame annotations
    for ann in current_frame_annotations:
        file = np.load(ann.result_path)
        results_to_draw.append((
            ann.sim_room_class.class_name,
            ann.sim_room_class.color,
            file["mask"],
            file["box"],
        ))

    # Add tracking results for current frame
    for cls_ in tracked_classes:
        class_id = cls_.id
        class_path = labeler.cal_rec.labeling_results_path / str(class_id)
        frame_idx = labeler.current_frame_idx
        result_path = class_path / f"{frame_idx}.npz"
        if not result_path.exists():
            continue

        file = np.load(result_path)
        results_to_draw.append((
            cls_.class_name,
            cls_.color,
            file["mask"],
            file["box"],
        ))

    return results_to_draw


def get_current_frame_overlay(db: Session) -> UInt8Array:
    labeler = require_labeler()
    results_to_draw = get_overlay_draw_data(db)

    # Draw masks first to avoid overlapping
    frame = labeler.current_frame.copy()
    for _, _, mask, box in results_to_draw:
        image_utils.draw_mask(frame, mask, box)

    # Draw bounding boxes
    for class_name, color, _, box in results_to_draw:
        image_utils.draw_box(frame, box, class_name, color)

    return frame


def get_timeline_context(
    request: Request,
    db: Session,
    polling: bool,
    frame_idx: int | None = None,
) -> LabelingTimelineContext:
    labeler = require_labeler()
    frame_idx = labeler.current_frame_idx if frame_idx is None else frame_idx
    selected_class_id = labeler.selected_class_id

    context = LabelingTimelineContext(
        request=request,
        current_frame_idx=frame_idx,
        frame_count=labeler.frame_count,
        selected_class_id=selected_class_id,
    )

    if selected_class_id != -1:
        sim_room_class = simrooms_repo.get_simroom_class(
            db=db,
            class_id=selected_class_id,
        )
        context.tracks = annotations_repo.get_tracks(labeler.current_class_results_path)
        context.selected_class_color = sim_room_class.color

    if (
        labeler.tracking_job is not None
        and labeler.tracking_job.class_id == selected_class_id
    ):
        context.tracking_progress = labeler.tracking_job.progress
        context.is_tracking = True

    if not polling:
        labeler.seek(frame_idx)
        context.update_canvas = True

    return context


def get_classes_context(
    request: Request,
    db: Session,
    selected_class_id: int,
) -> LabelingClassesContext:
    labeler = require_labeler()
    classes = simrooms_repo.get_simroom_classes(
        db=db,
        sim_room_id=labeler.cal_rec.sim_room_id,
    )

    if selected_class_id == -1 and len(classes) > 0:
        selected_class_id = classes[0].id

    labeler.selected_class_id = selected_class_id

    context = LabelingClassesContext(
        request=request,
        selected_class_id=selected_class_id,
        sim_room_id=labeler.cal_rec.sim_room_id,
        classes=classes,
    )

    return context


def get_annotations_context(request: Request, db: Session) -> LabelingAnnotationsContext:
    labeler = require_labeler()
    annotations = annotations_repo.get_annotations_by_class_id(
        db=db,
        calibration_recording_id=labeler.cal_rec.id,
        class_id=labeler.selected_class_id,
    )

    labeling_annotations = []
    for ann in annotations:
        file = np.load(ann.result_path)
        x1, y1, x2, y2 = file["box"]

        frame = get_frame_from_dir(ann.frame_idx, labeler.frames_path)
        frame_crop = frame[y1:y2, x1:x2]
        encoded_png = image_utils.encode_to_png(frame_crop)

        labeling_annotations.append(
            LabelingAnnotation(
                id=ann.id,
                frame_idx=ann.frame_idx,
                frame_crop=encoded_png,
            )
        )

    return LabelingAnnotationsContext(
        request=request,
        annotations=labeling_annotations,
    )


def find_closest_point_label(
    annotation: Annotation, point: tuple[int, int], max_distance: int = 1
) -> PointLabel:
    x, y = point
    points_labels = annotation.point_labels
    if not points_labels:
        return None

    closest_point_label = None
    min_distance = float("inf")
    for point_label in points_labels:
        distance = math.sqrt((point.x - x) ** 2 + (point.y - y) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_point_label = point_label

    if min_distance > max_distance:
        return None

    return closest_point_label


def post_annotation(
    db: Session,
    point: tuple[int, int],
    label: int,
    delete_point: bool = False,
):
    labeler = require_labeler()
    x, y = point
    selected_class_id = labeler.selected_class_id
    current_frame_idx = labeler.current_frame_idx
    calibration_id = labeler.cal_rec.id

    if selected_class_id == -1:
        raise NoClassSelectedError()

    # Check if there is an annotation for the current frame and class
    annotation = annotations_repo.get_annotation_by_frame_idx_and_class_id(
        db=db,
        calibration_recording_id=calibration_id,
        frame_idx=current_frame_idx,
        class_id=selected_class_id,
    )

    if delete_point and annotation is None:
        raise NotFoundError(
            f"Annotation not found for frame {current_frame_idx} and class {selected_class_id}"
        )
    elif delete_point:
        # Find the closest point to delete
        closest_point = find_closest_point_label(
            annotation=annotation,
            point=point,
        )
        if not closest_point:
            raise NotFoundError(
                f"Found no point to delete for frame {current_frame_idx}"
                f"and class {selected_class_id} at x={x}, y={y}"
            )
        annotations_repo.delete_point(db=db, id=closest_point.id)

    # If there's an annotation, retrieve its points and labels and delete it
    if annotation is not None:
        points = [(pl.x, pl.y) for pl in annotation.point_labels] + [(x, y)]
        labels = [pl.label for pl in annotation.point_labels] + [label]
        annotations_repo.delete_annotation(db=db, annotation_id=annotation.id)
    else:
        # If no annotation exists, create a new one with the point and label
        points = [(x, y)]
        labels = [label]

    # Predict image and save final results to the database
    mask, box = sam2_service.predict(
        predictor=labeler.image_predictor,
        points=points,
        points_labels=labels,
    )

    annotation = annotations_repo.create_annotation(
        db=db,
        calibration_recording_id=calibration_id,
        frame_idx=current_frame_idx,
        sim_room_class_id=selected_class_id,
        mask_base64=base64.b64encode(mask).decode("utf-8"),
        box_json=json.dumps(box),
    )
    annotations_repo.create_point_labels(
        db=db, annotation_id=annotation.id, points=points, labels=labels
    )


def start_tracking(db: Session) -> None:
    labeler = require_labeler()

    if labeler.tracking_job is not None:
        raise TrackingJobAlreadyRunningError()

    if labeler.selected_class_id == -1:
        raise NoClassSelectedError()

    annotations = annotations_repo.get_annotations_by_class_id(
        db=db,
        calibration_recording_id=labeler.cal_rec.id,
        class_id=labeler.selected_class_id,
    )
    annotations = [AnnotationDTO.from_orm(ann) for ann in annotations]

    labeler.tracking_job = TrackingJob(
        annotations=annotations,
        frames_path=labeler.frames_path,
        results_path=labeler.current_class_results_path,
        frame_count=labeler.frame_count,
        class_id=labeler.selected_class_id,
    )

    def job_runner():
        labeler.tracking_job.run()
        labeler.tracking_job = None

    threading.Thread(target=job_runner).start()


def post_settings(
    show_inactive_classes: bool,
) -> LabelingContext:
    labeler = require_labeler()
    labeler.show_inactive_classes = show_inactive_classes
