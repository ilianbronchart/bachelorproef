from dataclasses import asdict, dataclass

from g3pylib.recordings.recording import Recording


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
