import base64
import json
import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import aiohttp
import cv2
import numpy as np
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


def extract_frames_to_tmpdir(video_path):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"The file {video_path} does not exist.")

    temp_dir = tempfile.mkdtemp()
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    success, frame = video_capture.read()

    while success:
        frame_filename = os.path.join(temp_dir, f"{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        success, frame = video_capture.read()
        frame_count += 1

    video_capture.release()
    return temp_dir


def cv2_loadvideo(video_path: str) -> Generator[tuple[int, np.ndarray], None, None]:
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


def cv2_video_resolution(video_path: Path) -> tuple[int, int]:
    """
    Get the resolution of a video file using OpenCV.

    Args:
        video_path (str): Path to the video file.

    Returns:
        tuple[int, int]: The resolution of the video (height, width).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Video file not found: {video_path}")

    resolution = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    cap.release()
    return resolution


def cv2_video_fps(video_path: Path) -> float:
    """
    Get the frames per second (FPS) of a video file using OpenCV.

    Args:
        video_path (str): Path to the video file.

    Returns:
        float: The FPS of the video.
    """
    cap = cv2.VideoCapture(video_path)
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
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Video file not found: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def clamp(x, lower, upper):
    return max(lower, min(x, upper))


def base64_to_numpy(img: str):
    imgdata = base64.b64decode(img)
    nparr = np.frombuffer(imgdata, np.uint8)
    img_bgr = cv2.imdecode(nparr, flags=cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
