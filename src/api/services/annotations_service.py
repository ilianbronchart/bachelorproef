import json
import math

import numpy as np
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sqlalchemy.orm import Session

from src.aliases import UInt8Array
from src.api.models.pydantic import AnnotationDTO, PointLabelDTO
from src.api.repositories import annotations_repo
from src.api.services import sam2_service
from src.api.utils import image_utils


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


def get_annotations_by_frame_idx(
    db: Session, calibration_id: int, frame_idx: int
) -> list[AnnotationDTO]:
    annotations = annotations_repo.get_annotations_by_frame_idx(
        db=db,
        calibration_id=calibration_id,
        frame_idx=frame_idx,
    )

    return [AnnotationDTO.from_orm(annotation) for annotation in annotations]


def find_closest_point(
    points: list[tuple[int, int]], point: tuple[int, int], max_distance: int = 1
) -> int | None:
    x, y = point

    closest_index = None
    min_distance = float("inf")
    for i, p in enumerate(points):
        px, py = p
        distance = math.sqrt((px - x) ** 2 + (py - y) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_index = i

    if min_distance > max_distance:
        return None

    return closest_index


def create_annotation(
    db: Session,
    frame: UInt8Array,
    image_predictor: SAM2ImagePredictor,
    points: list[tuple[int, int]],
    labels: list[int],
    class_id: int,
    frame_idx: int,
    calibration_id: int,
) -> None:
    mask, box = sam2_service.predict(
        predictor=image_predictor,
        points=points,
        points_labels=labels,
    )
    mask = np.squeeze(mask)
    x1, y1, x2, y2 = box
    frame_crop = frame[y1:y2, x1:x2]

    annotation = annotations_repo.create_annotation(
        db=db,
        calibration_id=calibration_id,
        frame_idx=frame_idx,
        simroom_class_id=class_id,
        mask_base64=image_utils.encode_to_png(mask),
        frame_crop_base64=image_utils.encode_to_png(frame_crop),
        box_json=json.dumps([int(x1), int(y1), int(x2), int(y2)]),
    )
    annotations_repo.create_point_labels(
        db=db, annotation_id=annotation.id, points=points, labels=labels
    )


def update_annotation(
    db: Session,
    frame: UInt8Array,
    image_predictor: SAM2ImagePredictor,
    annotation_id: int,
    new_point: tuple[int, int],
    new_label: int,
    delete_point: bool,
) -> None:
    annotation = annotations_repo.get_annotation_by_id(
        db=db,
        annotation_id=annotation_id,
    )

    points = [(pl.x, pl.y) for pl in annotation.point_labels]
    labels = [pl.label for pl in annotation.point_labels]

    if delete_point:
        closest_point_index = find_closest_point(
            points=points,
            point=new_point,
        )
        if closest_point_index is not None:
            points.pop(closest_point_index)
            labels.pop(closest_point_index)
    else:
        points.append(new_point)
        labels.append(new_label)

    annotations_repo.delete_annotation(db=db, annotation_id=annotation.id)
    db.flush()

    if len(points) > 0:
        create_annotation(
            db=db,
            frame=frame,
            image_predictor=image_predictor,
            points=points,
            labels=labels,
            class_id=annotation.simroom_class_id,
            frame_idx=annotation.frame_idx,
            calibration_id=annotation.calibration_id,
        )


def post_annotation_point(
    db: Session,
    frame: UInt8Array,
    image_predictor: SAM2ImagePredictor,
    calibration_id: int,
    frame_idx: int,
    class_id: int,
    new_point: tuple[int, int],
    new_label: int,
    delete_point: bool,
) -> None:
    annotation = annotations_repo.get_annotation_by_frame_idx_and_class_id(
        db=db,
        calibration_id=calibration_id,
        frame_idx=frame_idx,
        class_id=class_id,
    )

    if annotation is not None:
        update_annotation(
            db=db,
            frame=frame,
            image_predictor=image_predictor,
            annotation_id=annotation.id,
            new_point=new_point,
            new_label=new_label,
            delete_point=delete_point,
        )
    else:
        create_annotation(
            db=db,
            frame=frame,
            image_predictor=image_predictor,
            points=[new_point],
            labels=[new_label],
            class_id=class_id,
            frame_idx=frame_idx,
            calibration_id=calibration_id,
        )
