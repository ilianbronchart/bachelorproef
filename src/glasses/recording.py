import os
from pathlib import Path

from g3pylib import connect_to_glasses

from src.config import DEFAULT_GLASSES_HOSTNAME, RECORDINGS_PATH
from src.glasses.domain import RecordingMetadata
from src.utils import download_file, load_json_files, save_json


def recording_exists(recording: RecordingMetadata, output_path: Path = RECORDINGS_PATH) -> bool:
    """Checks if a recording has already been downloaded."""
    return any(recording.uuid in file for file in os.listdir(output_path))


async def download_recording(recording: RecordingMetadata, output_path: Path = RECORDINGS_PATH) -> None:
    """
    Downloads the scene video, gazedata, and selected metadata from one Recording object.
    """
    if recording_exists(recording, output_path):
        raise ValueError(f"Recording {recording.uuid} already exists in {output_path}")

    try:
        # Download the scene video, gazedata, and metadata
        output_path.mkdir(parents=True, exist_ok=True)
        await download_file(recording.scene_video_url, output_path / f"{recording.uuid}.mp4")
        await download_file(recording.gaze_data_url, output_path / f"{recording.uuid}.tsv")
        save_json(recording.to_dict(), output_path / f"{recording.uuid}.json")
    except Exception as e:
        # Clean up created files if there is an error
        recordings = os.listdir(output_path)
        for file in recordings:
            if recording.uuid in file:
                (output_path / file).unlink()

        raise RuntimeError(f"Failed to download recording {recording.uuid}") from e


async def get_glasses_recordings(glasses_hostname: str = DEFAULT_GLASSES_HOSTNAME) -> list[RecordingMetadata]:
    """Retrieve metadata for all recordings on the glasses"""
    async with (
        connect_to_glasses.with_hostname(glasses_hostname, using_ip=True) as g3,
        g3.recordings.keep_updated_in_context(),
    ):
        recordings = g3.recordings.children
        return [await RecordingMetadata.from_recording(recording) for recording in recordings]


async def get_local_recordings(recordings_path: Path = RECORDINGS_PATH) -> list[RecordingMetadata]:
    """Retrieve metadata for all recordings in the local directory"""
    recordings_json = load_json_files(recordings_path)
    return [RecordingMetadata(**recording) for recording in recordings_json]
