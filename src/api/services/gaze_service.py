import json
from pathlib import Path

import numpy as np
import torch

from src.api.models.gaze import GazeData, GazePoint
from src.config import (
    RECORDINGS_PATH,
    TOBII_GLASSES_FPS,
    TOBII_GLASSES_RESOLUTION,
    VIEWED_RADIUS,
)
from src.utils import clamp


def mask_was_viewed(
    mask: torch.Tensor,
    gaze_position: tuple[float, float],
    viewed_radius: float = VIEWED_RADIUS,
) -> bool:
    """
    Check if the mask is at least partially within the viewed radius of the gaze point.
    The mask is assumed to be at the original frame size.

    Args:
        mask: A tensor containing a single mask of shape (H, W)
        gaze_position: Tuple (x, y) representing the gaze position.

    Returns:
        bool: True if part of the mask falls within the circular
              area defined by viewed_radius, False otherwise.
    """
    height, width = mask.shape
    device = mask.device

    # Create a coordinate grid for the mask.
    y_coords = torch.arange(0, height, device=device).view(-1, 1).repeat(1, width)
    x_coords = torch.arange(0, width, device=device).view(1, -1).repeat(height, 1)

    # Create the circular mask based on self.viewed_radius.
    dist_sq = (x_coords - gaze_position[0]) ** 2 + (y_coords - gaze_position[1]) ** 2
    circular_mask = (dist_sq <= viewed_radius**2).float()

    # Apply the circular mask to the input mask.
    overlapped_mask = mask * circular_mask
    return bool(overlapped_mask.sum() > 0)


def parse_gazedata_file(file_path: Path) -> list[GazeData]:
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")

    with file_path.open("r") as f:
        return [GazeData.from_dict(json.loads(line)) for line in f.readlines()]


def get_gaze_points(
    gaze_data: list[GazeData], resolution: tuple[int, int]
) -> list[GazePoint]:
    """
    Extract gaze points from a list of gaze data
    and denormalize them to the video resolution.
    Ignores gaze data with type MISSING.

    Args:
        gaze_data (List[GazeData]): List of gaze data objects.
        resolution (tuple[int, int]): Resolution of the video (height, width).

    Returns:
        List[GazePoint]: List of gaze points with denormalized coordinates.
    """

    gaze_points = []
    for data in gaze_data:
        if data.gaze2d is None:
            continue

        gaze_depth = None
        if data.gaze3d and data.eye_data_left and data.eye_data_right:
            eyeleft = data.eye_data_left
            eyeright = data.eye_data_right

            # Calculate gaze origin as the average of both eyes' gaze origins
            gaze_origin = (
                np.add(eyeleft.origin, eyeright.origin) / 2
                if (eyeleft and eyeright)
                else eyeleft.origin
                if eyeleft
                else eyeright.origin
            )

            # Calculate gaze depth as the distance from the gaze origin to the gaze point
            gaze_depth = np.sqrt(np.sum((data.gaze3d - gaze_origin) ** 2, axis=0))

        x = int(clamp(data.gaze2d[0], 0, 1) * resolution[1])
        y = int(clamp(data.gaze2d[1], 0, 1) * resolution[0])

        gaze_points.append(GazePoint(x, y, gaze_depth, data.timestamp))

    return gaze_points


def match_frames_to_gaze(
    frame_count: int, gaze_points: list[GazePoint], fps: float
) -> list[list[GazePoint]]:
    """
    Match video frames to their corresponding gaze points.
    The polling rate of gaze data is twice the fps of the video,
    so there are max two gaze points per frame.

    Args:
        num_frames (int): Number of video frames.
        gaze_points (List[GazePoint]): List of gaze points sorted by timestamp.
        fps (float): Frames per second of the video.

    Returns:
        List[FrameGazes]: List mapping each frame to its corresponding gaze points.
    """
    frame_gaze_mapping = []
    gaze_index = 0

    for frame_num in range(frame_count):
        next_frame_timestamp = (frame_num + 1) / fps
        frame_gazes = []

        while (
            gaze_index < len(gaze_points)
            and gaze_points[gaze_index].timestamp < next_frame_timestamp
        ):
            gaze_point = gaze_points[gaze_index]
            frame_gazes.append(gaze_point)
            gaze_index += 1

        frame_gaze_mapping.append(frame_gazes)

    for points in frame_gaze_mapping:
        if len(points) >= 3:
            print(
                f"Warning: Detected {len(points)} gaze points for a frame in the video. This is unexpected."
            )

    return frame_gaze_mapping


def get_gaze_point_per_frame(
    gaze_data_path: Path, resolution: tuple[int, int], frame_count: int, fps: float
) -> dict[int, GazePoint]:
    """
    Process gaze data and map frame indices to gaze points.

    Args:
        gaze_data_path (Path): Path to the gaze data file.
        resolution (tuple[int, int]): Video resolution as (height, width).
        frame_count (int): Number of frames in the video.
        fps (float): Frames per second of the video.

    Returns:
        dict[int, GazePoint]: Dictionary mapping frame indices
                              to their first valid gaze point.
    """
    gaze_data = parse_gazedata_file(gaze_data_path)
    gaze_points = get_gaze_points(gaze_data, resolution)
    frame_gaze_mapping = match_frames_to_gaze(
        frame_count=frame_count, gaze_points=gaze_points, fps=fps
    )

    gaze_point_per_frame = {
        frame_idx: gaze_points[0]
        for frame_idx, gaze_points in enumerate(frame_gaze_mapping)
        if len(gaze_points) > 0
    }

    return gaze_point_per_frame


def get_gaze_position_per_frame(
    recording_id: str,
    frame_count: int,
    resolution: tuple[int, int] = TOBII_GLASSES_RESOLUTION,
    fps: float = TOBII_GLASSES_FPS,
) -> dict[int, tuple[int, int]]:
    gaze_data_path = RECORDINGS_PATH / f"{recording_id}.tsv"
    gaze_point_per_frame = get_gaze_point_per_frame(
        gaze_data_path=gaze_data_path,
        resolution=resolution,
        frame_count=frame_count,
        fps=fps,
    )
    gaze_position_per_frame = {
        frame_idx: (gaze_point.x, gaze_point.y)
        for frame_idx, gaze_point in gaze_point_per_frame.items()
    }
    return gaze_position_per_frame
