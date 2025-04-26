import asyncio

from g3pylib import connect_to_glasses
from g3pylib.recordings.recording import Recording as GlassesRecording
from src.config import DEFAULT_GLASSES_HOSTNAME
from src.db import Recording


async def is_connected(glasses_hostname: str = DEFAULT_GLASSES_HOSTNAME) -> bool:
    return True

    try:

        async def connect():
            async with connect_to_glasses.with_hostname(glasses_hostname, using_ip=True):
                return True

        # Use asyncio.wait_for to apply a timeout
        return await asyncio.wait_for(connect(), timeout=1)  # Timeout after 2 seconds
    except asyncio.TimeoutError:
        return False


async def get_battery_level(glasses_hostname: str = DEFAULT_GLASSES_HOSTNAME) -> float:
    return 82
    try:
        async with connect_to_glasses.with_hostname(
            glasses_hostname, using_ip=True
        ) as g3:
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


async def get_recordings(
    glasses_hostname: str = DEFAULT_GLASSES_HOSTNAME,
) -> list[Recording]:
    return [
        Recording(
            uuid="946cffb5-a018-4b7a-bf67-ab7a172af12c",
            visible_name="Test recording",
            participant="Test participant",
            created="2023-10-01T12:00:00",
            duration="00:01:00",
            folder_name="test_folder",
            scene_video_url="https://example.com/scene_video.mp4",
            gaze_data_url="https://example.com/gaze_data.csv",
        ),
        Recording(
            uuid="3bd86d9b-06e1-40fa-b587-fdba559a344c",
            visible_name="Another recording",
            participant="Another participant",
            created="2023-10-02T12:00:00",
            duration="00:02:00",
            folder_name="another_folder",
            scene_video_url="https://example.com/another_scene_video.mp4",
            gaze_data_url="https://example.com/another_gaze_data.csv",
        ),
        Recording(
            uuid="39f5164f-873d-4d6b-be6b-e1d5db79c02a",
            visible_name="Third recording",
            participant="Third participant",
            created="2023-10-03T12:00:00",
            duration="00:03:00",
            folder_name="third_folder",
            scene_video_url="https://example.com/third_scene_video.mp4",
            gaze_data_url="https://example.com/third_gaze_data.csv",
        ),
        Recording(
            uuid="56f64c07-8066-4def-a965-df05616d56a6",
            visible_name="Fourth recording",
            participant="Fourth participant",
            created="2023-10-04T12:00:00",
            duration="00:04:00",
            folder_name="fourth_folder",
            scene_video_url="https://example.com/fourth_scene_video.mp4",
            gaze_data_url="https://example.com/fourth_gaze_data.csv",
        ),
    ]

    """Retrieve metadata for all recordings on the glasses"""
    async with (
        connect_to_glasses.with_hostname(glasses_hostname, using_ip=True) as g3,
        g3.recordings.keep_updated_in_context(),
    ):
        recordings = g3.recordings.children
        return [await recording_from_glasses(recording) for recording in recordings]


async def get_recording(
    uuid: str, glasses_hostname: str = DEFAULT_GLASSES_HOSTNAME
) -> Recording:
    """Retrieve metadata for recording by its UUID"""
    async with (
        connect_to_glasses.with_hostname(glasses_hostname, using_ip=True) as g3,
        g3.recordings.keep_updated_in_context(),
    ):
        glasses_recording = g3.recordings.get_recording(uuid)
        return await recording_from_glasses(glasses_recording)
