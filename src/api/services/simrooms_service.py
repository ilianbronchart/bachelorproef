import tempfile
from pathlib import Path

from sqlalchemy.orm import Session

from src.api.models.pydantic import CalibrationRecordingDTO, SimRoomClassDTO, SimRoomDTO
from src.api.repositories import simrooms_repo
from src.config import RECORDINGS_PATH
from src.utils import extract_frames_to_dir


def get_class_id_to_name_map(db: Session, simroom_id: int) -> dict[int, str]:
    """
    Get a mapping of class IDs to class names.
    """
    classes = simrooms_repo.get_simroom_classes(db, simroom_id)
    return {simroom_class.id: simroom_class.class_name for simroom_class in classes}


def get_tracked_classes(db: Session, calibration_id: int) -> list[SimRoomClassDTO]:
    """
    Get all classes that have annotations for a given calibration recording.
    """
    classes = simrooms_repo.get_tracked_classes(db, calibration_id)
    return [SimRoomClassDTO.from_orm(simroom_class) for simroom_class in classes]


def create_simroom(db: Session, name: str) -> SimRoomDTO:
    """
    Create a new sim room.
    """
    simroom = simrooms_repo.create_simroom(db, name=name)
    return SimRoomDTO.from_orm(simroom)


def get_simroom_class(db: Session, class_id: int) -> SimRoomClassDTO:
    """
    Get a sim room class by its ID.
    """
    simroom_class = simrooms_repo.get_simroom_class(db, class_id)
    return SimRoomClassDTO.from_orm(simroom_class)


def get_simroom_classes(db: Session, simroom_id: int) -> list[SimRoomClassDTO]:
    """
    Get all sim room classes for a given sim room ID.
    """
    classes = simrooms_repo.get_simroom_classes(db, simroom_id)
    return [SimRoomClassDTO.from_orm(simroom_class) for simroom_class in classes]


def get_calibration_recording(
    db: Session, calibration_id: int
) -> CalibrationRecordingDTO:
    """
    Get a calibration recording by its ID.
    """
    calibration_recording = simrooms_repo.get_calibration_recording(db, calibration_id)
    return CalibrationRecordingDTO.from_orm(calibration_recording)


def extract_tmp_frames(
    recording_id: str,
    recordings_path: Path = RECORDINGS_PATH,
):
    recording_path = recordings_path / f"{recording_id}.mp4"
    tmp_frames_dir = tempfile.TemporaryDirectory()
    tmp_frames_path = Path(tmp_frames_dir.name)
    extract_frames_to_dir(
        video_path=recording_path,
        frames_path=tmp_frames_path,
        print_output=False,
    )
    frames = sorted(tmp_frames_path.glob("*.jpg"), key=lambda x: int(x.stem))

    return frames, tmp_frames_dir
