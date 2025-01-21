from dataclasses import asdict, dataclass

from g3pylib.recordings.recording import Recording
from datetime import datetime


@dataclass
class RecordingMetadata:
    uuid: str
    visible_name: str
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
            created=(await recording.get_created()).isoformat(),
            duration=str(await recording.get_duration()),
            folder_name=await recording.get_folder(),
            scene_video_url=await recording.get_scenevideo_url(),
            gaze_data_url=await recording.get_gazedata_url(),
        )

    def to_dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}

    def get_formatted(self):
        formatted_dict = self.to_dict()
        formatted_dict["duration"] = self._format_duration(self.duration)
        formatted_dict["created"] = self._format_datetime(self.created)
        return formatted_dict

    def _format_duration(self, duration: str) -> str:
        hours, minutes, seconds = map(float, duration.split(':'))
        total_seconds = int(hours * 3600 + minutes * 60 + seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        remaining_seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"

    def _format_datetime(self, datetime_str: str) -> str:
        dt = datetime.fromisoformat(datetime_str)
        return dt.strftime("%d/%m/%y at %I:%M %p")
