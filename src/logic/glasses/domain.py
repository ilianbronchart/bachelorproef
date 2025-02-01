import base64
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum

import numpy as np
from g3pylib.recordings.recording import Recording


@dataclass
class RecordingMetadata:
    uuid: str
    visible_name: str
    participant: str
    created: str
    duration: str
    folder_name: str
    scene_video_url: str
    gaze_data_url: str

    @staticmethod
    async def from_recording(recording: Recording) -> "RecordingMetadata":
        return RecordingMetadata(
            uuid=recording.uuid,
            visible_name=await recording.get_visible_name(),
            participant=await RecordingMetadata.parse_participant(recording),
            created=(await recording.get_created()).isoformat(),
            duration=str(await recording.get_duration()),
            folder_name=await recording.get_folder(),
            scene_video_url=await recording.get_scenevideo_url(),
            gaze_data_url=await recording.get_gazedata_url(),
        )

    @staticmethod
    async def parse_participant(recording: Recording) -> str:
        participant_bytes = base64.b64decode(await recording.meta_lookup("participant"))
        participant_json = participant_bytes.decode("utf-8")
        parsed_data = json.loads(participant_json)
        return parsed_data["name"] or "N/A"

    def to_dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}

    def get_formatted(self):
        formatted_dict = self.to_dict()
        formatted_dict["duration"] = self._format_duration(self.duration)
        dt = datetime.fromisoformat(self.created)
        formatted_dict["created"] = dt.strftime("%d/%m/%y at %I:%M %p")
        return formatted_dict

    def _format_duration(self, duration: str) -> str:
        hours, minutes, seconds = map(float, duration.split(":"))
        total_seconds = int(hours * 3600 + minutes * 60 + seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        remaining_seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


@dataclass
class GazePoint:
    x: int
    y: int
    depth: float
    timestamp: float

    @property
    def position(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class EyeGazeData:
    origin: np.ndarray
    direction: np.ndarray
    pupil_diameter: float


class GazeDataType(Enum):
    GAZE: str = "gaze"
    MISSING: str = "missing"


@dataclass
class GazeData:
    type: str
    timestamp: float
    gaze2d: tuple[float, float] | None
    gaze3d: tuple[float, float, float] | None
    eye_data_left: EyeGazeData | None
    eye_data_right: EyeGazeData | None

    @staticmethod
    def from_dict(data: dict) -> "GazeData":
        gaze_data = data["data"]

        if len(gaze_data.keys()) == 0:
            # No gaze data available for this timestamp
            return GazeData(
                type=GazeDataType.MISSING,
                timestamp=data["timestamp"],
                gaze2d=None,
                gaze3d=None,
                eye_data_left=None,
                eye_data_right=None,
            )

        left_eye = gaze_data.get("eyeleft", {})
        right_eye = gaze_data.get("eyeright", {})

        return GazeData(
            type=GazeDataType.GAZE,
            timestamp=data["timestamp"],
            gaze2d=tuple(gaze_data["gaze2d"]),
            gaze3d=tuple(gaze_data["gaze3d"]),
            eye_data_left=EyeGazeData(
                origin=np.array(left_eye["gazeorigin"]),
                direction=np.array(left_eye["gazedirection"]),
                pupil_diameter=left_eye["pupildiameter"],
            )
            if left_eye
            else None,
            eye_data_right=EyeGazeData(
                origin=np.array(right_eye["gazeorigin"]),
                direction=np.array(right_eye["gazedirection"]),
                pupil_diameter=right_eye["pupildiameter"],
            )
            if right_eye
            else None,
        )
