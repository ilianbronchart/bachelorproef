import asyncio
from datetime import datetime
from pathlib import Path

from g3pylib import connect_to_glasses
from sqlalchemy.orm import Session

from src.api.exceptions import (
    GlassesDisconnectedError,
    InternalError,
    NotFoundError,
    RecordingAlreadyExistsError,
)
from src.api.models.pydantic import RecordingDTO
from src.api.repositories import recordings_repo
from src.api.services import recordings_service
from src.config import DEBUG_MODE, DEFAULT_GLASSES_HOSTNAME, RECORDINGS_PATH
from src.utils import download_file

if DEBUG_MODE:
    MOCK_RECORDINGS = [
        RecordingDTO(
            id="946cffb5-a018-4b7a-bf67-ab7a172af12c",
            visible_name="Test recording",
            participant="Test participant",
            created=datetime.fromisoformat("2023-10-01T12:00:00"),
            duration="00:01:00",
        ),
        RecordingDTO(
            id="3bd86d9b-06e1-40fa-b587-fdba559a344c",
            visible_name="Another recording",
            participant="Another participant",
            created=datetime.fromisoformat("2023-10-02T12:00:00"),
            duration="00:02:00",
        ),
        RecordingDTO(
            id="39f5164f-873d-4d6b-be6b-e1d5db79c02a",
            visible_name="Third recording",
            participant="Third participant",
            created=datetime.fromisoformat("2023-10-03T12:00:00"),
            duration="00:03:00",
        ),
        RecordingDTO(
            id="56f64c07-8066-4def-a965-df05616d56a6",
            visible_name="Fourth recording",
            participant="Fourth participant",
            created=datetime.fromisoformat("2023-10-04T12:00:00"),
            duration="00:04:00",
        ),
    ]


async def is_connected(glasses_hostname: str = DEFAULT_GLASSES_HOSTNAME) -> bool:
    if DEBUG_MODE:
        return True

    try:

        async def connect() -> bool:
            async with connect_to_glasses.with_hostname(glasses_hostname, using_ip=True):
                return True

        # Use asyncio.wait_for to apply a timeout
        return await asyncio.wait_for(connect(), timeout=1)  # Timeout after 2 seconds
    except asyncio.TimeoutError:
        return False


async def get_battery_level(glasses_hostname: str = DEFAULT_GLASSES_HOSTNAME) -> float:
    if DEBUG_MODE:
        return 82
    try:
        async with connect_to_glasses.with_hostname(
            glasses_hostname, using_ip=True
        ) as g3:
            return round((await g3.system.battery.get_level()) * 100)
    except asyncio.TimeoutError:
        return -1


async def get_recordings(
    glasses_hostname: str = DEFAULT_GLASSES_HOSTNAME,
) -> list[RecordingDTO]:
    """Retrieve metadata for all recordings on the glasses"""
    if DEBUG_MODE:
        return MOCK_RECORDINGS

    async with (
        connect_to_glasses.with_hostname(glasses_hostname, using_ip=True) as g3,
        g3.recordings.keep_updated_in_context(),
    ):
        recordings = g3.recordings.children
        return [
            await RecordingDTO.from_glasses_recording(recording)
            for recording in recordings
        ]


async def get_recording(
    recording_id: str, glasses_hostname: str = DEFAULT_GLASSES_HOSTNAME
) -> RecordingDTO:
    """Retrieve metadata for recording by its ID"""
    try:
        async with (
            connect_to_glasses.with_hostname(glasses_hostname, using_ip=True) as g3,
            g3.recordings.keep_updated_in_context(),
        ):
            glasses_rec = g3.recordings.get_recording(recording_id)
            return await RecordingDTO.from_glasses_recording(glasses_rec)
    except TimeoutError as e:
        raise GlassesDisconnectedError(
            f"Failed to connect to glasses at {glasses_hostname}"
        ) from e
    except KeyError as e:
        raise NotFoundError(
            f"Recording with ID {recording_id} not found on glasses"
        ) from e


async def download_recording(
    db: Session,
    recording_id: str,
    recordings_path: Path = RECORDINGS_PATH,
    glasses_hostname: str = DEFAULT_GLASSES_HOSTNAME,
) -> None:
    if not recordings_path.exists():
        raise InternalError(f"Recordings path {recordings_path} does not exist")

    if recordings_service.recording_is_complete(db, recording_id):
        raise RecordingAlreadyExistsError(
            f"Recording {recording_id} already exists in local directory"
        )

    async with (
        connect_to_glasses.with_hostname(glasses_hostname, using_ip=True) as g3,
        g3.recordings.keep_updated_in_context(),
    ):
        glasses_rec = g3.recordings.get_recording(recording_id)
        scene_video_url = await glasses_rec.get_scenevideo_url()
        gaze_data_url = await glasses_rec.get_gazedata_url()
        video_path = recordings_path / f"{glasses_rec.uuid}.mp4"
        gaze_data_path = recordings_path / f"{glasses_rec.uuid}.tsv"

        try:
            await download_file(scene_video_url, video_path)
            await download_file(gaze_data_url, gaze_data_path)
        except Exception as e:
            # Clean up created files if there is an error
            video_path.unlink(missing_ok=True)
            gaze_data_path.unlink(missing_ok=True)
            raise InternalError(
                f"Failed to download recording {recording_id}: {e}"
            ) from e

        rec_dto = await RecordingDTO.from_glasses_recording(glasses_rec)
        recordings_repo.create(
            db=db,
            recording_id=rec_dto.id,
            visible_name=rec_dto.visible_name,
            participant=rec_dto.participant,
            created=rec_dto.created.isoformat(),
            duration=rec_dto.duration,
        )
