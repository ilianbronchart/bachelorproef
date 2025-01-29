from pathlib import Path
from typing import List, Tuple

import numpy as np
from src.core.utils import clamp
from src.logic.glasses.domain import GazeData, GazeDataType, GazePoint
import json

def parse_gazedata_file(file_path: Path) -> List[GazeData]:
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")

    with file_path.open("r") as f:
        return [GazeData.from_dict(json.loads(line)) for line in f.readlines()]
    
def get_gaze_points(gaze_data: List[GazeData], resolution: tuple[int, int]) -> List[GazePoint]:
    """
    Extract gaze points from a list of gaze data and denormalize them to the video resolution.

    Args:
        gaze_data (List[GazeData]): List of gaze data objects.
        resolution (tuple[int, int]): Resolution of the video (height, width).
    
    Returns:
        List[GazePoint]: List of gaze points with denormalized coordinates.
    """

    valid_gazedata = [data for data in gaze_data if data.type != GazeDataType.MISSING]

    gaze_points = []
    for data in valid_gazedata:
        eyeleft = data.eye_data_left
        eyeright = data.eye_data_right

        # Calculate gaze origin as the average of both eyes' gaze origins
        gaze_origin = (
            np.add(eyeleft.origin, eyeright.origin) / 2 if (eyeleft and eyeright)
            else eyeleft.origin if eyeleft
            else eyeright.origin
        )

        # Calculate gaze depth as the distance from the gaze origin to the gaze point
        gaze_depth = np.sqrt(np.sum((data.gaze3d-gaze_origin)**2, axis=0))
        x = int(clamp(data.gaze2d[0], 0, 1) * resolution[1])
        y = int(clamp(data.gaze2d[1], 0, 1) * resolution[0])

        gaze_points.append(GazePoint(x, y, gaze_depth, data.timestamp))

    return gaze_points

def match_frames_to_gaze(
    num_frames: int, 
    gaze_points: List[GazePoint], 
    fps: float
) -> List[List[GazePoint]]:
    """
    Match video frames to their corresponding gaze points.
    The polling rate of gaze data is twice the fps of the video, so there are max two gaze points per frame.

    Args:
        num_frames (int): Number of video frames.
        gaze_points (List[GazePoint]): List of gaze points sorted by timestamp.
        fps (float): Frames per second of the video.

    Returns:
        List[FrameGazes]: List mapping each frame to its corresponding gaze points.
    """
    frame_gaze_mapping = []
    gaze_index = 0

    for frame_num in range(num_frames):
        next_frame_timestamp = (frame_num + 1) / fps
        frame_gazes = []

        while gaze_index < len(gaze_points) and gaze_points[gaze_index].timestamp < next_frame_timestamp:
            gaze_point = gaze_points[gaze_index]
            frame_gazes.append(gaze_point)
            gaze_index += 1
        
        frame_gaze_mapping.append(frame_gazes)

    gaze_counts = set(sorted([len(points) for points in frame_gaze_mapping], reverse=True))
    if 3 in gaze_counts:
        raise Warning(f"Detected 3 gaze points for a frame in the video. This is unexpected.")

    return frame_gaze_mapping