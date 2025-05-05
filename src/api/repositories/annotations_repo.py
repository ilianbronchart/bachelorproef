from pathlib import Path

from sqlalchemy.orm import Session
from src.api.exceptions import NotFoundError
from src.api.models.db import Annotation, PointLabel


def get_annotations_by_frame_idx(
    db: Session,
    calibration_id: int,
    frame_idx: int,
) -> list[Annotation]:
    annotations = (
        db.query(Annotation)
        .filter(
            Annotation.calibration_id == calibration_id,
            Annotation.frame_idx == frame_idx,
        )
        .all()
    )
    return annotations


def get_annotation_by_frame_idx_and_class_id(
    db: Session,
    calibration_id: int,
    frame_idx: int,
    class_id: int,
) -> Annotation | None:
    annotation = (
        db.query(Annotation)
        .filter(
            Annotation.calibration_id == calibration_id,
            Annotation.frame_idx == frame_idx,
            Annotation.simroom_class_id == class_id,
        )
        .first()
    )
    return annotation


def get_annotations_by_class_id(
    db: Session,
    calibration_id: int,
    class_id: int,
) -> list[Annotation]:
    annotations = (
        db.query(Annotation)
        .filter(
            Annotation.calibration_id == calibration_id,
            Annotation.simroom_class_id == class_id,
        )
        .all()
    )
    return annotations


def get_annotation_by_id(
    db: Session,
    annotation_id: int,
) -> Annotation:
    annotation = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    if not annotation:
        raise NotFoundError(f"Annotation with id {annotation_id} not found")
    return annotation


def create_annotation(
    db: Session,
    calibration_id: int,
    frame_idx: int,
    simroom_class_id: int,
    mask_base64: str,
    frame_crop_base64: str,
    box_json: str,
) -> Annotation:
    annotation = Annotation(
        calibration_id=calibration_id,
        frame_idx=frame_idx,
        simroom_class_id=simroom_class_id,
        mask_base64=mask_base64,
        frame_crop_base64=frame_crop_base64,
        box_json=box_json,
    )
    db.add(annotation)
    db.flush()
    db.refresh(annotation)
    return annotation


def delete_annotation(db: Session, annotation_id: int) -> None:
    annotation = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    if not annotation:
        raise NotFoundError(f"Annotation with id {annotation_id} not found")
    db.delete(annotation)
    db.flush()


def delete_point_label(db: Session, point_label_id: int) -> None:
    point = db.query(PointLabel).filter(PointLabel.id == point_label_id).first()
    if not point:
        raise NotFoundError(f"Point with id {point_label_id} not found")
    db.delete(point)


def create_point_labels(
    db: Session,
    annotation_id: int,
    points: list[tuple[int, int]],
    labels: list[int],
) -> None:
    for (x, y), label in zip(points, labels, strict=False):
        point_label = PointLabel(
            annotation_id=annotation_id,
            x=x,
            y=y,
            label=label,
        )
        db.add(point_label)


def get_tracks(class_tracking_results_path: Path) -> list[tuple[int, int]]:
    """
    Get all tracks for the given class labeling results path.
    Returns a list of tuples, where each tuple contains the
    start and end frame index of a track.
    """
    if not class_tracking_results_path.exists():
        return []

    results = list(class_tracking_results_path.glob("*.npz"))
    results_frame_idx = sorted(int(result.stem) for result in results)

    if not results_frame_idx:
        return []

    tracks = []
    start = results_frame_idx[0]
    prev = start

    for frame in results_frame_idx[1:]:
        # Check if the current frame is consecutive to the previous one
        if frame == prev + 1:
            prev = frame  # Continue the current track
        else:
            # End the current track and start a new one
            tracks.append((start, prev))
            start = frame
            prev = frame

    # Add the final track
    tracks.append((start, prev))
    return tracks
