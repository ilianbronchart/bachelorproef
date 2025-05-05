from sqlalchemy.orm import Session
from src.api.exceptions import NotFoundError
from src.api.models.db import Recording


def delete(db: Session, recording_id: str) -> None:
    rec = db.query(Recording).filter(Recording.id == recording_id).first()
    if rec is None:
        raise NotFoundError(f"Recording with id {recording_id} not found")

    rec.video_path.unlink(missing_ok=True)
    rec.gaze_data_path.unlink(missing_ok=True)
    db.delete(rec)


def get(db: Session, recording_id: str) -> Recording:
    """Get a recording by its ID"""
    rec = db.query(Recording).filter(Recording.id == recording_id).first()
    if rec is None:
        raise NotFoundError(f"Recording with id {recording_id} not found")
    return rec


def get_all(db: Session) -> list[Recording]:
    return db.query(Recording).all()


def create(
    db: Session,
    recording_id: str,
    visible_name: str,
    participant: str,
    created: str,
    duration: str,
) -> None:
    db_recording = Recording(
        id=recording_id,
        visible_name=visible_name,
        participant=participant,
        created=created,
        duration=duration,
    )
    db.add(db_recording)
