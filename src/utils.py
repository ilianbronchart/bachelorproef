import base64
import json
import random
import shutil
import subprocess
from collections.abc import Generator
from pathlib import Path

import aiohttp
import cv2
import numpy as np
from fastapi import Request

from src.aliases import UInt8Array


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


def clamp(x: float, lower: float, upper: float) -> float:
    return max(lower, min(x, upper))


def base64_to_numpy(img: str) -> UInt8Array:
    imgdata = base64.b64decode(img)
    nparr = np.frombuffer(imgdata, np.uint8)
    img_bgr = cv2.imdecode(nparr, flags=cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)


def generate_pleasant_color() -> str:
    """Generate a random color with moderate saturation and brightness."""
    hue = random.random()  # noqa: S311
    saturation = random.uniform(0.4, 0.6)  # noqa: S311
    brightness = random.uniform(0.6, 0.8)  # noqa: S311

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


def iter_frames_dir(frames_path: Path) -> Generator[tuple[int, UInt8Array], None, None]:
    frames = [frame for frame in frames_path.iterdir() if ".jpg" in frame.suffix]

    if len(frames) == 0:
        raise FileNotFoundError(f"No frames found in {frames_path}")

    for frame_path in sorted(frames_path.iterdir()):
        frame_idx = int(frame_path.stem)
        frame = cv2.imread(str(frame_path)).astype(np.uint8)
        yield frame_idx, frame


def extract_frames_to_dir(
    video_path: Path, frames_path: Path, print_output: bool = False
) -> None:
    if not video_path.name.endswith(".mp4"):
        raise ValueError(f"Video file must be in MP4 format, got: {video_path.name}")

    # Delete any existing frames
    for file in frames_path.iterdir():
        file.unlink()

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise FileNotFoundError("ffmpeg executable not found in PATH")

    # Validate ffmpeg_path to ensure it's not tampered or injected
    if not Path(ffmpeg_path).exists() or Path(ffmpeg_path).name != "ffmpeg":
        raise ValueError("Invalid ffmpeg executable path")

    # Conditionally redirect output based on the print_output argument
    stdout = None if print_output else subprocess.DEVNULL
    stderr = None if print_output else subprocess.DEVNULL

    # TODO: fix noqa here
    subprocess.run(  # noqa: S603
        [
            ffmpeg_path,
            "-i",
            str(video_path),
            "-q:v",
            "2",
            "-start_number",
            "0",
            f"{frames_path!s}/%05d.jpg",
        ],
        check=True,
        stdout=stdout,
        stderr=stderr,
        shell=False,
    )


def get_frame_from_dir(frame_idx: int, frames_path: Path) -> UInt8Array:
    frame_path = frames_path / f"{frame_idx:05}.jpg"
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame {frame_idx} not found in {frames_path}")

    return cv2.imread(str(frame_path)).astype(np.uint8)


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Convert a hex color string to a BGR tuple for OpenCV."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)
