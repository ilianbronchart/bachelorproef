import shutil
from pathlib import Path

from sqlalchemy.orm import Session
from src.api.exceptions import NotFoundError
from src.api.models.db import (
    Annotation,
    CalibrationRecording,
    Recording,
    SimRoom,
    SimRoomClass,
)
from src.api.models.pydantic import SimRoomClassDTO, SimRoomDTO
from src.config import LABELING_RESULTS_PATH


def get_simroom(db: Session, id: str) -> SimRoomDTO:
    """Get a recording by its ID"""
    sim_room = db.query(SimRoom).filter(SimRoom.id == id).first()
    if sim_room is None:
        raise NotFoundError(f"SimRoom with id {id} not found")
    return SimRoomDTO.from_orm(sim_room)


def get_all_simrooms(db: Session) -> list[SimRoomDTO]:
    """Get all sim rooms"""
    sim_rooms = db.query(SimRoom).all()
    return [SimRoomDTO.from_orm(sim_room) for sim_room in sim_rooms]


def create_simroom(
    db: Session,
    name: str,
) -> SimRoomDTO:
    """Create a new sim room"""
    sim_room = SimRoom(name=name)
    db.add(sim_room)
    db.flush()
    db.refresh(sim_room)
    return SimRoomDTO.from_orm(sim_room)


def delete_simroom(db: Session, id: str) -> None:
    """Delete a sim room"""
    sim_room = db.query(SimRoom).filter(SimRoom.id == id).first()
    if sim_room is None:
        raise NotFoundError(f"SimRoom with id {id} not found")

    db.delete(sim_room)


def create_simroom_class(
    db: Session, simroom_id: int, class_name: str
) -> SimRoomClassDTO:
    """Create a new sim room class"""
    simroom_class = SimRoomClass(class_name=class_name, simroom_id=simroom_id)
    db.add(simroom_class)
    db.flush()
    db.refresh(simroom_class)
    return SimRoomClassDTO.from_orm(simroom_class)


def delete_simroom_class(db: Session, class_id: int) -> None:
    """Delete a sim room class"""
    simroom_class = db.query(SimRoomClass).filter(SimRoomClass.id == class_id).first()
    if simroom_class is None:
        raise NotFoundError(f"SimRoomClass with id {class_id} not found")

    # For the tracking results of each calibration recording
    # remove the results for the class id
    for labeling_results in LABELING_RESULTS_PATH.iterdir():
        tracking_results_path: Path = labeling_results / str(class_id)
        if tracking_results_path.exists():
            shutil.rmtree(tracking_results_path)

    db.delete(simroom_class)


def get_calibration_recording(
    db: Session,
    calibration_id: int = None,
    simroom_id: int = None,
    recording_id: str = None,
) -> CalibrationRecording | None:
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
                f"CalibrationRecording with simroom_id {simroom_id} and recording_id {recording_id} not found"
            )
        return calibration_recording
    else:
        raise ValueError("Invalid arguments for get_calibration_recording")


def add_calibration_recording(db: Session, simroom_id: int, recording_id: str) -> None:
    """Add a calibration recording to a sim room"""
    sim_room = db.query(SimRoom).filter(SimRoom.id == simroom_id).first()
    if sim_room is None:
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


def get_simroom_class(db: Session, class_id: int) -> SimRoomClassDTO:
    """Get a sim room class by its ID"""
    simroom_class = db.query(SimRoomClass).filter(SimRoomClass.id == class_id).first()
    if simroom_class is None:
        raise NotFoundError(f"SimRoomClass with id {class_id} not found")
    return SimRoomClassDTO.from_orm(simroom_class)


def get_simroom_classes(db: Session, simroom_id: int) -> list[SimRoomClass]:
    """Get all classes for a sim room"""
    sim_room = db.query(SimRoom).filter(SimRoom.id == simroom_id).first()
    if sim_room is None:
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
