import base64
import json
import random
from collections.abc import Generator
from pathlib import Path
from typing import cast

import aiohttp
import cv2
import numpy as np
import numpy.typing as npt
from fastapi import Request


async def download_file(url: str, target_path: Path) -> None:
    """
    Downloads a file from a URL and saves it to a local path asynchronously.

    Args:
        url (str): The URL of the file to download.
        local_path (str): The local path where the file will be saved.

    Returns:
        None

    Raises:
        aiohttp.ClientError: If there is an error during the HTTP request.
        IOError: If there is an error writing the file to disk.
    """
    with target_path.open("wb") as fd:
        async with aiohttp.ClientSession() as session, session.get(url) as resp:
            async for chunk in resp.content.iter_chunked(1024 * 64):
                fd.write(chunk)


def save_json(data: dict[str, str], target_path: Path) -> None:
    with target_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json_files(target_path: Path) -> list[dict[str, str]]:
    files = [file for file in target_path.iterdir() if file.suffix == ".json"]
    return [json.load(file.open(encoding="utf-8")) for file in files]


def is_hx_request(request: Request) -> bool:
    return request.headers.get("hx-request") == "true"


def cv2_loadvideo(video_path: str) -> Generator[tuple[int, cv2.typing.MatLike], None, None]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Video file not found: {video_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            # read over
            break

        yield frame_idx, frame
        frame_idx += 1

    cap.release()


def cv2_video_resolution(video_path: Path, flip: bool = False) -> tuple[int, int]:
    """
    Get the resolution of a video file using OpenCV.

    Args:
        video_path (str): Path to the video file.
        flip: Whether to flip the resolution (width, height) instead of (height, width).

    Returns:
        tuple[int, int]: The resolution of the video (height, width).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Video file not found: {video_path}")

    resolution = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    cap.release()

    if flip:
        resolution = (resolution[1], resolution[0])
    return resolution


def cv2_video_fps(video_path: Path) -> float:
    """
    Get the frames per second (FPS) of a video file using OpenCV.

    Args:
        video_path (str): Path to the video file.

    Returns:
        float: The FPS of the video.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Video file not found: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def cv2_video_frame_count(video_path: Path) -> int:
    """
    Get the frame count of a video file using OpenCV.

    Args:
        video_path (str): Path to the video file.

    Returns:
        int: The number of frames in the video.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Video file not found: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def cv2_get_frame(video_path: Path, frame_idx: int) -> npt.NDArray[np.uint8]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Video file not found: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx < 0 or frame_idx >= frame_count:
        cap.release()
        raise IndexError(f"Frame index {frame_idx} is out of range, total frames: {frame_count}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError(f"Could not read frame at index {frame_idx}")

    cap.release()
    return cast(npt.NDArray[np.uint8], frame)


def clamp(x: float, lower: float, upper: float) -> float:
    return max(lower, min(x, upper))


def base64_to_numpy(img: str):
    imgdata = base64.b64decode(img)
    nparr = np.frombuffer(imgdata, np.uint8)
    img_bgr = cv2.imdecode(nparr, flags=cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def generate_pleasant_color() -> str:
    """Generate a random color with moderate saturation and brightness."""
    hue = random.random()  # Random hue (0-1)
    saturation = random.uniform(0.4, 0.6)  # Moderate saturation
    brightness = random.uniform(0.6, 0.8)  # Moderate to high brightness

    # Convert HSV to RGB
    h = hue * 6
    i = int(h)
    f = h - i
    p = brightness * (1 - saturation)
    q = brightness * (1 - saturation * f)
    t = brightness * (1 - saturation * (1 - f))

    if i % 6 == 0:
        r, g, b = brightness, t, p
    elif i % 6 == 1:
        r, g, b = q, brightness, p
    elif i % 6 == 2:
        r, g, b = p, brightness, t
    elif i % 6 == 3:
        r, g, b = p, q, brightness
    elif i % 6 == 4:
        r, g, b = t, p, brightness
    else:
        r, g, b = brightness, p, q

    # Convert to hex
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
