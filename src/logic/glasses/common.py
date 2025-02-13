import asyncio

from g3pylib import connect_to_glasses
from g3pylib.recordings.recording import Recording as GlassesRecording
from src.config import DEFAULT_GLASSES_HOSTNAME
from src.db import Recording


async def is_connected(glasses_hostname: str = DEFAULT_GLASSES_HOSTNAME) -> bool:
    try:

        async def connect():
            async with connect_to_glasses.with_hostname(glasses_hostname, using_ip=True):
                return True

        # Use asyncio.wait_for to apply a timeout
        return await asyncio.wait_for(connect(), timeout=1)  # Timeout after 2 seconds
    except asyncio.TimeoutError:
        return False


async def get_battery_level(glasses_hostname: str = DEFAULT_GLASSES_HOSTNAME) -> float:
    try:
        async with connect_to_glasses.with_hostname(glasses_hostname, using_ip=True) as g3:
            return round((await g3.system.battery.get_level()) * 100)
    except asyncio.TimeoutError:
        return -1


async def recording_from_glasses(glasses_recording: GlassesRecording) -> Recording:
    """Create a recording object from a glasses recording object"""
    recording = Recording(
        uuid=glasses_recording.uuid,
        visible_name=await glasses_recording.get_visible_name(),
        participant=await Recording.parse_participant(glasses_recording),
        created=(await glasses_recording.get_created()).isoformat(),
        duration=str(await glasses_recording.get_duration()),
        folder_name=await glasses_recording.get_folder(),
        scene_video_url=await glasses_recording.get_scenevideo_url(),
        gaze_data_url=await glasses_recording.get_gazedata_url(),
    )
    return recording


async def get_recordings(glasses_hostname: str = DEFAULT_GLASSES_HOSTNAME) -> list[Recording]:
    """Retrieve metadata for all recordings on the glasses"""
    async with (
        connect_to_glasses.with_hostname(glasses_hostname, using_ip=True) as g3,
        g3.recordings.keep_updated_in_context(),
    ):
        recordings = g3.recordings.children
        return [await recording_from_glasses(recording) for recording in recordings]


async def get_recording(uuid: str, glasses_hostname: str = DEFAULT_GLASSES_HOSTNAME) -> Recording:
    """Retrieve metadata for recording by its UUID"""
    async with (
        connect_to_glasses.with_hostname(glasses_hostname, using_ip=True) as g3,
        g3.recordings.keep_updated_in_context(),
    ):
        glasses_recording = g3.recordings.get_recording(uuid)
        return await recording_from_glasses(glasses_recording)
