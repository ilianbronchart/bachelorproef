import shutil
from pathlib import Path

from sqlalchemy.orm import Session
from src.api.exceptions import NotFoundError
from src.api.models.db import (
    CalibrationRecording,
    Recording,
    SimRoom,
    SimRoomClass,
)
from src.api.models.pydantic import SimRoomDTO
from src.config import TRACKING_RESULTS_PATH


def get_simroom(db: Session, simroom_id: int) -> SimRoomDTO:
    """Get a recording by its ID"""
    simroom = db.query(SimRoom).filter(SimRoom.id == simroom_id).first()
    if simroom is None:
        raise NotFoundError(f"SimRoom with id {simroom_id} not found")
    return SimRoomDTO.from_orm(simroom)


def get_all_simrooms(db: Session) -> list[SimRoomDTO]:
    """Get all sim rooms"""
    simrooms = db.query(SimRoom).all()
    return [SimRoomDTO.from_orm(simroom) for simroom in simrooms]


def create_simroom(
    db: Session,
    name: str,
) -> SimRoom:
    """Create a new sim room"""
    simroom = SimRoom(name=name)
    db.add(simroom)
    db.flush()
    db.refresh(simroom)
    return simroom


def delete_simroom(db: Session, simroom_id: int) -> None:
    """Delete a sim room"""
    simroom = db.query(SimRoom).filter(SimRoom.id == simroom_id).first()
    if simroom is None:
        raise NotFoundError(f"SimRoom with id {simroom_id} not found")

    db.delete(simroom)


def create_simroom_class(db: Session, simroom_id: int, class_name: str) -> SimRoomClass:
    """Create a new sim room class"""
    simroom_class = SimRoomClass(class_name=class_name, simroom_id=simroom_id)
    db.add(simroom_class)
    db.flush()
    db.refresh(simroom_class)
    return simroom_class


def delete_simroom_class(db: Session, class_id: int) -> None:
    """Delete a sim room class"""
    simroom_class = db.query(SimRoomClass).filter(SimRoomClass.id == class_id).first()
    if simroom_class is None:
        raise NotFoundError(f"SimRoomClass with id {class_id} not found")

    # For the tracking results of each calibration recording
    # remove the results for the class id
    for labeling_results in TRACKING_RESULTS_PATH.iterdir():
        tracking_results_path: Path = labeling_results / str(class_id)
        if tracking_results_path.exists():
            shutil.rmtree(tracking_results_path)

    db.delete(simroom_class)


def get_calibration_recording(
    db: Session,
    calibration_id: int | None = None,
    simroom_id: int | None = None,
    recording_id: str | None = None,
) -> CalibrationRecording:
    if calibration_id is not None:
        calibration_recording = (
            db.query(CalibrationRecording)
            .filter(CalibrationRecording.id == calibration_id)
            .first()
        )
        if calibration_recording is None:
            raise NotFoundError(
                f"CalibrationRecording with id {calibration_id} not found"
            )
        return calibration_recording
    elif simroom_id is not None and recording_id is not None:
        calibration_recording = (
            db.query(CalibrationRecording)
            .filter(
                CalibrationRecording.simroom_id == simroom_id,
                CalibrationRecording.recording_id == recording_id,
            )
            .first()
        )
        if calibration_recording is None:
            raise NotFoundError(
                f"CalibrationRecording with simroom_id {simroom_id}"
                f"and recording_id {recording_id} not found"
            )
        return calibration_recording
    else:
        raise ValueError("Invalid arguments for get_calibration_recording")


def add_calibration_recording(db: Session, simroom_id: int, recording_id: str) -> None:
    """Add a calibration recording to a sim room"""
    simroom = db.query(SimRoom).filter(SimRoom.id == simroom_id).first()
    if simroom is None:
        raise NotFoundError(f"SimRoom with id {simroom_id} not found")

    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if recording is None:
        raise NotFoundError(f"Recording with id {recording_id} not found")

    calibration_recording = CalibrationRecording(
        simroom_id=simroom_id,
        recording_id=recording_id,
    )
    db.add(calibration_recording)
    db.flush()
    db.refresh(calibration_recording)

    # Create the labeling results paths
    calibration_recording.tracking_results_path.mkdir(parents=True, exist_ok=True)


def delete_calibration_recording(db: Session, calibration_id: int) -> None:
    """Delete a calibration recording"""
    calibration_recording = (
        db.query(CalibrationRecording)
        .filter(CalibrationRecording.id == calibration_id)
        .first()
    )
    if calibration_recording is None:
        raise NotFoundError(f"CalibrationRecording with id {calibration_id} not found")

    # Remove labeling results for the calibration recording
    if calibration_recording.tracking_results_path.exists():
        shutil.rmtree(calibration_recording.tracking_results_path)

    db.delete(calibration_recording)


def get_simroom_class(db: Session, class_id: int) -> SimRoomClass:
    """Get a sim room class by its ID"""
    simroom_class = db.query(SimRoomClass).filter(SimRoomClass.id == class_id).first()
    if simroom_class is None:
        raise NotFoundError(f"SimRoomClass with id {class_id} not found")
    return simroom_class


def get_simroom_classes(db: Session, simroom_id: int) -> list[SimRoomClass]:
    """Get all classes for a sim room"""
    simroom = db.query(SimRoom).filter(SimRoom.id == simroom_id).first()
    if simroom is None:
        raise NotFoundError(f"SimRoom with id {simroom_id} not found")

    classes = db.query(SimRoomClass).filter(SimRoomClass.simroom_id == simroom_id).all()
    return classes


def get_classes_by_ids(db: Session, class_ids: list[int]) -> list[SimRoomClass]:
    """Get all classes for a sim room"""
    classes = db.query(SimRoomClass).filter(SimRoomClass.id.in_(class_ids)).all()
    return classes


def get_tracked_classes(db: Session, calibration_id: int) -> list[SimRoomClass]:
    """
    Get all classes that have annotations for a given calibration recording.
    """
    cal_rec = get_calibration_recording(db, calibration_id=calibration_id)
    result_paths = cal_rec.tracking_result_paths
    result_paths = [path for path in result_paths if len(list(path.iterdir())) > 0]
    class_ids = [int(path.stem) for path in result_paths]
    return get_classes_by_ids(db, class_ids)


def get_tracking_result_paths(
    db: Session, calibration_id: int, class_id: int
) -> list[Path]:
    """
    Get all tracking result paths for a given calibration recording and class ID.
    """
    cal_rec = get_calibration_recording(db, calibration_id=calibration_id)
    result_paths = cal_rec.tracking_result_paths
    result_paths = [path for path in result_paths if path.stem == str(class_id)]
    if not result_paths:
        raise NotFoundError(
            f"No tracking results found for calibration "
            f"ID {calibration_id} and class ID {class_id}"
        )

    return list(result_paths[0].iterdir())
