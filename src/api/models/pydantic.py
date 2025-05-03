import base64
import json
from datetime import datetime
from pathlib import Path

from g3pylib.recordings.recording import Recording as GlassesRecording
from pydantic import BaseModel

from src.api.models.db import Recording as DBRecording
from src.config import RECORDINGS_PATH


class RecordingDTO(BaseModel):
    uuid: str
    visible_name: str
    participant: str
    created: str
    duration: str

    # pre‑formatted strings so the template doesn’t have to repeat logic
    @property
    def formatted_duration(self) -> str:
        parts = self.duration.split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(float(parts[2]))
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @property
    def formatted_created(self) -> str:
        return datetime.fromisoformat(str(self.created)).strftime("%d/%m/%y at %I:%M %p")

    @property
    def video_path(self) -> Path:
        return RECORDINGS_PATH / f"{self.uuid}.mp4"

    @property
    def gaze_data_path(self) -> Path:
        return RECORDINGS_PATH / f"{self.uuid}.tsv"

    class Config:
        orm_mode = True

    @classmethod
    async def parse_participant(recording: GlassesRecording) -> str:
        participant_bytes = base64.b64decode(await recording.meta_lookup("participant"))
        participant_json = participant_bytes.decode("utf-8")
        parsed_data = json.loads(participant_json)
        return parsed_data["name"] or "N/A"

    @classmethod
    def from_orm(cls, rec: DBRecording) -> "RecordingDTO":
        return cls(
            uuid=rec.uuid,
            visible_name=rec.visible_name,
            participant=rec.participant,
            created=rec.created,
            duration=rec.duration,
        )

    @classmethod
    async def from_glasses_recording(
        cls, glasses_recording: GlassesRecording
    ) -> "RecordingDTO":
        return cls(
            uuid=glasses_recording.uuid,
            visible_name=await glasses_recording.get_visible_name(),
            participant=await cls.parse_participant(glasses_recording),
            created=(await glasses_recording.get_created()).isoformat(),
            duration=str(await glasses_recording.get_duration()),
        )
