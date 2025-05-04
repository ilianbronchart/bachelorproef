import base64
import json
import math

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sqlalchemy.orm import Session

from src.api.exceptions import NotFoundError
from src.api.models.db import Annotation, PointLabel
from src.api.models.pydantic import AnnotationDTO, PointLabelDTO
from src.api.repositories import annotations_repo
from src.api.services import sam2_service


class PointLabelWithClassID(PointLabelDTO):
    class_id: int


def get_point_labels(
    db: Session, calibration_id: int, frame_idx: int
) -> list[PointLabelWithClassID]:
    annotations = annotations_repo.get_annotations_by_frame_idx(
        db=db,
        calibration_id=calibration_id,
        frame_idx=frame_idx,
    )

    point_labels = []
    for annotation in annotations:
        for point_label in annotation.point_labels:
            dto = PointLabelDTO.from_orm(point_label)  # or from_orm if your config allows
            with_class = PointLabelWithClassID.model_validate(
                dto.model_dump() | {"class_id": annotation.simroom_class_id}
            )
            point_labels.append(with_class)

    return point_labels


def get_annotations_by_class_id(
    db: Session, calibration_id: int, class_id: int
) -> list[AnnotationDTO]:
    annotations = annotations_repo.get_annotations_by_class_id(
        db=db,
        calibration_id=calibration_id,
        class_id=class_id,
    )

    return [AnnotationDTO.from_orm(annotation) for annotation in annotations]


def _find_closest_point_label(
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


def create_or_update_annotation(
    db: Session,
    image_predictor: SAM2ImagePredictor,
    point: tuple[int, int],
    label: int,
    class_id: int,
    frame_idx: int,
    calibration_id: int,
    delete_point: bool,
) -> None:
    x, y = point

    annotation = annotations_repo.get_annotation_by_frame_idx_and_class_id(
        db=db,
        calibration_id=calibration_id,
        frame_idx=frame_idx,
        class_id=class_id,
    )

    if delete_point and annotation is None:
        raise NotFoundError(
            f"Annotation not found for frame {frame_idx} and class {class_id}"
        )
    elif delete_point:
        # Find the closest point to delete
        closest_point = _find_closest_point_label(
            annotation=annotation,
            point=point,
        )
        if not closest_point:
            raise NotFoundError(
                f"Found no point to delete for frame {frame_idx}"
                f"and class {class_id} at x={x}, y={y}"
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
        predictor=image_predictor,
        points=points,
        points_labels=labels,
    )

    annotation = annotations_repo.create_annotation(
        db=db,
        calibration_id=calibration_id,
        frame_idx=frame_idx,
        simroom_class_id=class_id,
        mask_base64=base64.b64encode(mask).decode("utf-8"),
        box_json=json.dumps(box),
    )
    annotations_repo.create_point_labels(
        db=db, annotation_id=annotation.id, points=points, labels=labels
    )
