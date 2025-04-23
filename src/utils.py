import base64
import json
import random
import shutil
import subprocess
from collections.abc import Generator
from pathlib import Path
from typing import cast

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


def cv2_itervideo(
    video_path: str,
) -> Generator[tuple[int, cv2.typing.MatLike], None, None]:
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

    resolution = (
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )
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


def cv2_get_frame(video_path: Path, frame_idx: int) -> UInt8Array:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Video file not found: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx < 0 or frame_idx >= frame_count:
        cap.release()
        raise IndexError(
            f"Frame index {frame_idx} is out of range, total frames: {frame_count}"
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError(f"Could not read frame at index {frame_idx}")

    cap.release()
    return cast(UInt8Array, frame)


def clamp(x: float, lower: float, upper: float) -> float:
    return max(lower, min(x, upper))


def base64_to_numpy(img: str):
    imgdata = base64.b64decode(img)
    nparr = np.frombuffer(imgdata, np.uint8)
    img_bgr = cv2.imdecode(nparr, flags=cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


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
        frame = cv2.imread(str(frame_path))
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

    return cv2.imread(str(frame_path))


def encode_to_png(image: UInt8Array) -> str:
    """Encode an image to PNG format and return as base64 string."""
    ret, encoded_img = cv2.imencode(".png", image)
    if not ret:
        raise ValueError("Failed to encode image to PNG")
    return base64.b64encode(encoded_img.tobytes()).decode("utf-8")


def hex_to_bgr(hex_color: str) -> tuple:
    """Convert a hex color string to a BGR tuple for OpenCV."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


def draw_annotation_on_frame(
    frame_img: np.ndarray,
    mask: np.ndarray | None,
    box: tuple[int, int, int, int],
    class_color_hex: str,
    class_name: str,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Annotate a frame with a mask overlay, bounding box, and class name.

    Parameters:
      frame_img (np.ndarray): The original frame image in BGR format.
      mask (np.ndarray): The mask corresponding to the bounding box region.
                         Its shape should be (H, W) or (1, H, W).
      box (tuple): A tuple (x1, y1, x2, y2) representing the bounding box.
      class_color_hex (str): The class color in hex format (e.g. "#FF00AA").
      class_name (str): The class name text.
      alpha (float): The opacity for the white overlay (default 0.5).

    Returns:
      np.ndarray: The annotated frame image.
    """
    # Unpack bounding box coordinates
    x1, y1, x2, y2 = box
    class_color = hex_to_bgr(class_color_hex)

    if mask is not None:
        # Squeeze mask if it has an extra dimension (e.g., shape (1, H, W) -> (H, W))
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        # Ensure mask is boolean
        if mask.dtype != bool:
            mask = mask.astype(bool)

        # Extract the ROI corresponding to the bounding box
        roi = frame_img[y1:y2, x1:x2]

        # Blend white (255,255,255) into the ROI where mask is True (50% opacity by default)
        roi[mask] = (roi[mask].astype(float) * (1 - alpha) + 255 * alpha).astype(np.uint8)
        # Update the frame with the modified ROI
        frame_img[y1:y2, x1:x2] = roi

    # Draw the bounding box using the converted BGR color
    cv2.rectangle(frame_img, (x1, y1), (x2, y2), class_color, 2)

    # Prepare the text for the class name
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text = class_name
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    # Determine the text position: above the bounding box if there's room, otherwise below
    text_top = y1 - text_h - 4 if y1 - text_h - 4 > 0 else y1 + text_h + 4

    # Draw a filled rectangle as background for the text
    cv2.rectangle(
        frame_img, (x1, text_top), (x1 + text_w, text_top + text_h + 4), class_color, -1
    )
    # Put the class name text on top of the rectangle
    cv2.putText(
        frame_img,
        text,
        (x1, text_top + text_h + 2),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    return frame_img
