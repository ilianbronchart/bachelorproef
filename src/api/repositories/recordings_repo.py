from pathlib import Path

from sqlalchemy.orm import Session
from src.api.exceptions import NotFoundError
from src.api.models.db import Recording
from src.config import RECORDINGS_PATH


def delete(db: Session, id: str, recordings_path: Path = RECORDINGS_PATH):
    rec = db.query(Recording).filter(Recording.id == id).first()
    if rec is None:
        raise NotFoundError(f"Recording with id {id} not found")

    rec.video_path.unlink(missing_ok=True)
    rec.gaze_data_path.unlink(missing_ok=True)
    db.delete(rec)


def get(db: Session, id: str) -> Recording:
    """Get a recording by its ID"""
    rec = db.query(Recording).filter(Recording.id == id).first()
    if rec is None:
        raise NotFoundError(f"Recording with id {id} not found")
    return rec


def get_all(db: Session) -> list[Recording]:
    return db.query(Recording).all()


def create(
    db: Session,
    id: str,
    visible_name: str,
    participant: str,
    created: str,
    duration: str,
    folder_name: str,
) -> Recording:
    db_recording = Recording(
        id=id,
        visible_name=visible_name,
        participant=participant,
        created=created,
        duration=duration,
        folder_name=folder_name,
    )
    db.add(db_recording)
