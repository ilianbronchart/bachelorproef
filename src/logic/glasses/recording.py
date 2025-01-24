import os
from pathlib import Path

from g3pylib import connect_to_glasses
from src.core.config import DEFAULT_GLASSES_HOSTNAME, RECORDINGS_PATH
from src.core.utils import download_file, load_json_files, save_json
from src.logic.glasses.domain import RecordingMetadata


def recording_exists(uuid: str, output_path: Path = RECORDINGS_PATH) -> bool:
    """Checks if all files of a recording exist in the output path"""
    return all((output_path / f"{uuid}.{ext}").exists() for ext in ["mp4", "tsv", "json"])


async def download_recording(recording: RecordingMetadata, output_path: Path = RECORDINGS_PATH) -> None:
    """
    Downloads the scene video, gazedata, and selected metadata from one Recording object.
    """
    if recording_exists(recording.uuid, output_path):
        raise ValueError(f"Recording {recording.uuid} already exists in {output_path}")

    # Clean up any existing files for the recording (in case of previous partial download)
    delete_local_recording(recording.uuid, output_path)

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


async def get_recording(uuid: str, glasses_hostname: str = DEFAULT_GLASSES_HOSTNAME) -> RecordingMetadata:
    """Retrieve metadata for recording by its UUID"""
    async with (
        connect_to_glasses.with_hostname(glasses_hostname, using_ip=True) as g3,
        g3.recordings.keep_updated_in_context(),
    ):
        recording = g3.recordings.get_recording(uuid)
        return await RecordingMetadata.from_recording(recording)


async def get_local_recordings(recordings_path: Path = RECORDINGS_PATH) -> list[RecordingMetadata]:
    """Retrieve metadata for all recordings in the local directory"""
    recordings_json = load_json_files(recordings_path)
    return [RecordingMetadata(**recording) for recording in recordings_json]


def delete_local_recording(uuid: str, recordings_path: Path = RECORDINGS_PATH) -> None:
    """Delete a recording from the local directory"""
    recordings = [file for file in os.listdir(recordings_path) if uuid in file]
    for file in recordings:
        (recordings_path / file).unlink()


def clean_local_recordings(recordings_path: Path = RECORDINGS_PATH) -> None:
    "Delete all partially downloaded recordings from the local directory"
    uuids = {file.stem for file in recordings_path.iterdir()}
    for uuid in uuids:
        if not recording_exists(uuid, recordings_path):
            delete_local_recording(uuid, recordings_path)
