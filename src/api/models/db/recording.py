import base64
import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Union

from g3pylib.recordings.recording import Recording as GlassesRecording
from sqlalchemy import String
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship

from src.config import RECORDINGS_PATH
from src.api.db import Base, engine
from src.utils import download_file

# Workaround for circular imports due to type references
if TYPE_CHECKING:
    from .calibration import CalibrationRecording


class Recording(Base):
    __tablename__ = "recordings"

    uuid: Mapped[str] = mapped_column(String, primary_key=True)
    visible_name: Mapped[str] = mapped_column(String)
    participant: Mapped[str] = mapped_column(String)
    created: Mapped[str] = mapped_column(String)
    duration: Mapped[str] = mapped_column(String)
    folder_name: Mapped[str] = mapped_column(String)
    scene_video_url: Mapped[str] = mapped_column(String)
    gaze_data_url: Mapped[str] = mapped_column(String)

    calibration_recordings: Mapped[list["CalibrationRecording"]] = relationship(
        "CalibrationRecording", back_populates="recording"
    )

    @property
    def video_path(self) -> Path:
        return RECORDINGS_PATH / f"{self.uuid}.mp4"

    @property
    def gaze_data_path(self) -> Path:
        return RECORDINGS_PATH / f"{self.uuid}.tsv"

    @property
    def formatted_created(self) -> str:
        return datetime.fromisoformat(str(self.created)).strftime("%d/%m/%y at %I:%M %p")

    @property
    def formatted_duration(self) -> str:
        parts = self.duration.split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(float(parts[2]))
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @staticmethod
    async def parse_participant(recording: GlassesRecording) -> str:
        participant_bytes = base64.b64decode(await recording.meta_lookup("participant"))
        participant_json = participant_bytes.decode("utf-8")
        parsed_data = json.loads(participant_json)
        return parsed_data["name"] or "N/A"

    @staticmethod
    def get(uuid: str) -> Union["Recording", None]:
        with Session(engine) as session:
            return session.query(Recording).filter(Recording.uuid == uuid).first()

    @staticmethod
    def get_all() -> list["Recording"]:
        with Session(engine) as session:
            return session.query(Recording).all()

    @staticmethod
    def clean_recordings(recordings_path: Path = RECORDINGS_PATH) -> None:
        with Session(engine) as session:
            recordings = session.query(Recording).all()

        for recording in recordings:
            if not recording.is_complete(recordings_path):
                recording.remove(recordings_path)

        # Delete files whose stem is not a valid recording uuid
        valid_uuids = {str(recording.uuid) for recording in recordings}
        for file in recordings_path.iterdir():
            if file.is_file() and file.stem not in valid_uuids:
                file.unlink()

    def remove(self, recordings_path: Path = RECORDINGS_PATH) -> None:
        recordings = [
            file for file in os.listdir(recordings_path) if str(self.uuid) in file
        ]
        for file in recordings:
            (recordings_path / file).unlink()

        with Session(engine) as session:
            session.delete(self)
            session.commit()

    async def download(self, recordings_path: Path = RECORDINGS_PATH) -> None:
        if self.is_complete(recordings_path):
            raise ValueError(f"Recording {self.uuid} already exists in {recordings_path}")

        if not recordings_path.exists():
            raise FileNotFoundError(f"Recordings path {recordings_path} does not exist")

        try:
            await download_file(
                str(self.scene_video_url), recordings_path / f"{self.uuid}.mp4"
            )
            await download_file(
                str(self.gaze_data_url), recordings_path / f"{self.uuid}.tsv"
            )

            with Session(engine) as session:
                session.add(self)
                session.commit()
        except Exception as e:
            # Clean up created files if there is an error
            self.remove(recordings_path)
            raise RuntimeError(f"Failed to download recording {self.uuid}") from e

    def is_complete(self, recordings_path: Path = RECORDINGS_PATH) -> bool:
        """Checks if all files of a recording exist in the output recordings path"""
        is_in_db = self.get(str(self.uuid)) is not None
        return is_in_db and all(
            (recordings_path / f"{self.uuid}.{ext}").exists() for ext in ["mp4", "tsv"]
        )
