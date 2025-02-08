import base64
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Union

from g3pylib.recordings.recording import Recording as GlassesRecording
from sqlalchemy import Column, String
from sqlalchemy.orm import Session, relationship

from src.config import RECORDINGS_PATH
from src.core.utils import download_file
from src.db.db import Base, engine


class Recording(Base):
    __tablename__ = "recordings"

    uuid = Column(String, primary_key=True)
    visible_name = Column(String)
    participant = Column(String)
    created = Column(String)
    duration = Column(String)
    folder_name = Column(String)
    scene_video_url = Column(String)
    gaze_data_url = Column(String)

    # One recording can be linked to multiple calibration recordings
    calibration_recordings = relationship("CalibrationRecording", back_populates="recording")

    def to_dict(self) -> dict[str, str]:
        return {
            "uuid": self.uuid,
            "visible_name": self.visible_name,
            "participant": self.participant,
            "created": self.created,
            "duration": self.duration,
            "folder_name": self.folder_name,
            "scene_video_url": self.scene_video_url,
            "gaze_data_url": self.gaze_data_url,
        }

    @staticmethod
    async def from_glasses(glasses_recording: GlassesRecording) -> "Recording":
        return Recording(
            uuid=glasses_recording.uuid,
            visible_name=await glasses_recording.get_visible_name(),
            participant=await Recording.parse_participant(glasses_recording),
            created=(await glasses_recording.get_created()).isoformat(),
            duration=str(await glasses_recording.get_duration()),
            folder_name=await glasses_recording.get_folder(),
            scene_video_url=await glasses_recording.get_scenevideo_url(),
            gaze_data_url=await glasses_recording.get_gazedata_url(),
        )

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
    def get_all() -> List["Recording"]:
        with Session(engine) as session:
            return session.query(Recording).all()

    @staticmethod
    def clean_recordings(recordings_path: Path = RECORDINGS_PATH):
        with Session(engine) as session:
            recordings = session.query(Recording).all()
        for recording in recordings:
            if not recording.is_complete(recordings_path):
                recording.remove(recordings_path)

    def get_formatted(self):
        formatted_dict = self.to_dict()
        formatted_dict["duration"] = self._format_duration(self.duration)
        formatted_dict["created"] = datetime.fromisoformat(self.created).strftime("%d/%m/%y at %I:%M %p")
        return formatted_dict

    def _format_duration(self, duration: str) -> str:
        hours, minutes, seconds = map(float, duration.split(":"))
        total_seconds = int(hours * 3600 + minutes * 60 + seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        remaining_seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"

    def remove(self, recordings_path: Path = RECORDINGS_PATH):
        recordings = [file for file in os.listdir(recordings_path) if self.uuid in file]
        for file in recordings:
            (recordings_path / file).unlink()

        with Session(engine) as session:
            session.delete(self)
            session.commit()

    async def download(self, recordings_path: Path = RECORDINGS_PATH):
        if self.is_complete(recordings_path):
            raise ValueError(f"Recording {self.uuid} already exists in {recordings_path}")

        if not recordings_path.exists():
            raise FileNotFoundError(f"Recordings path {recordings_path} does not exist")

        try:
            await download_file(self.scene_video_url, recordings_path / f"{self.uuid}.mp4")
            await download_file(self.gaze_data_url, recordings_path / f"{self.uuid}.tsv")

            with Session(engine) as session:
                session.add(self)
                session.commit()
        except Exception as e:
            # Clean up created files if there is an error
            self.remove(recordings_path)
            raise RuntimeError(f"Failed to download recording {self.uuid}") from e

    def is_complete(self, recordings_path: Path = RECORDINGS_PATH) -> bool:
        """Checks if all files of a recording exist in the output recordings path"""
        is_in_db = self.get(self.uuid) is not None
        return is_in_db and all((recordings_path / f"{self.uuid}.{ext}").exists() for ext in ["mp4", "tsv"])
