from pathlib import Path

from sqlalchemy.orm import Session

from src.api.models.pydantic import RecordingDTO
from src.api.repositories import recordings_repo
from src.config import RECORDINGS_PATH


def get(db: Session, recording_id: str) -> RecordingDTO:
    """Get a recording by its ID"""
    rec = recordings_repo.get(db, recording_id)
    return RecordingDTO.from_orm(rec)


def get_all(db: Session) -> list[RecordingDTO]:
    """Get all recordings"""
    recordings = recordings_repo.get_all(db)
    return [RecordingDTO.from_orm(rec) for rec in recordings]


def recording_is_complete(
    db: Session, recording_id: str, recordings_path: Path = RECORDINGS_PATH
) -> bool:
    """
    Checks if a db entry exists for a recording
    and if all its files exist in the recordings path
    """
    is_complete = get(db, recording_id) is not None
    is_complete = is_complete and all(
        (recordings_path / f"{recording_id}.{ext}").exists() for ext in ["mp4", "tsv"]
    )
    return is_complete


def clean_recordings(db: Session, recordings_path: Path = RECORDINGS_PATH) -> None:
    recordings = get_all(db)
    for recording in recordings:
        if not (recording.video_path.exists() and recording.gaze_data_path.exists()):
            recordings_repo.delete(db, recording.id)

    # Delete files whose stem is not a valid recording id
    valid_ids = {str(recording.id) for recording in recordings}
    for file in recordings_path.iterdir():
        if file.is_file() and file.stem not in valid_ids:
            file.unlink()
