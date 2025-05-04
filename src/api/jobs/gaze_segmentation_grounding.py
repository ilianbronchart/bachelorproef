import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from src.api.services.gaze_service import (
    get_gaze_points,
    match_frames_to_gaze,
    parse_gazedata_file,
)
from src.config import FAST_SAM_CHECKPOINT, GAZE_FOVEA_FOV, TOBII_FOV_X
from src.utils import cv2_video_fps, cv2_video_frame_count, cv2_video_resolution
from torchvision.ops import masks_to_boxes
from torchvision.transforms import InterpolationMode
from ultralytics import FastSAM


class GazeSegmentationJob:
    def __init__(
        self,
        video_path: Path,
        gaze_data_path: Path,
        results_path: Path,
        fovea_fov: float = GAZE_FOVEA_FOV,
        fov_x: float = TOBII_FOV_X,
        checkpoint_path: Path = FAST_SAM_CHECKPOINT,
        output_video_path: Path | None = None,
    ):
        self.video_path = video_path
        self.gaze_data_path = gaze_data_path
        self.fovea_fov = fovea_fov
        self.fov_x = fov_x

        # Set up the results directory.
        self.results_path = results_path
        if self.results_path.exists():
            shutil.rmtree(self.results_path, ignore_errors=True)
            self.results_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)

        # Load the FastSAM model.
        self.model = FastSAM(checkpoint_path)

        # Video properties.
        self.resolution = cv2_video_resolution(self.video_path)
        self.aspect_ratio = self.resolution[1] / self.resolution[0]  # W / H
        self.fps = cv2_video_fps(self.video_path)
        self.viewed_radius = int((self.fovea_fov / self.fov_x) * self.resolution[1])
        self.frame_count = cv2_video_frame_count(self.video_path)

        # Set up the output video.
        if output_video_path is not None:
            self.video_result = cv2.VideoWriter(
                str(output_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (self.resolution[1], self.resolution[0]),
            )
        else:
            self.video_result = None

        # Parse gaze data.
        self.gaze_data = parse_gazedata_file(self.gaze_data_path)
        self.gaze_points = get_gaze_points(self.gaze_data, self.resolution)

        # Map frame indexes to gaze points.
        self.frame_gaze_mapping = match_frames_to_gaze(
            self.frame_count, self.gaze_points, self.fps
        )

    def get_gaze_position(self, frame_idx: int) -> tuple[int, int] | None:
        """
        Get the gaze position for a frame index.
        """
        gaze_points = self.frame_gaze_mapping[frame_idx]
        if len(gaze_points) == 0:
            return None
        return gaze_points[0].position

    def mask_too_large(self, mask: torch.Tensor) -> bool:
        """
        Check if the mask area is less than or equal to 30% of the frame area.

        Args:
            mask: A tensor containing a single mask of shape (H, W)

        Returns:
            bool: True if the mask's area is less than or equal to
                  30% of the frame area, False otherwise.
        """
        height, width = mask.shape
        frame_area = height * width
        max_mask_area = 0.1 * frame_area

        mask_area = mask.sum()
        return bool(mask_area >= max_mask_area)

    def mask_was_viewed(
        self, mask: torch.Tensor, gaze_position: tuple[float, float]
    ) -> bool:
        """
        Check if the mask is at least partially within the
        viewed radius of the gaze point.
        The mask is assumed to be at the original frame size.

        Args:
            mask: A tensor containing a single mask of shape (H, W)
            gaze_position: Tuple (x, y) representing the gaze position.

        Returns:
            bool: True if part of the mask falls within the circular
                  area defined by self.viewed_radius, False otherwise.
        """
        height, width = mask.shape
        device = mask.device

        # Create a coordinate grid for the mask.
        y_coords = torch.arange(0, height, device=device).view(-1, 1).repeat(1, width)
        x_coords = torch.arange(0, width, device=device).view(1, -1).repeat(height, 1)
        dist_sq = (x_coords - gaze_position[0]) ** 2 + (y_coords - gaze_position[1]) ** 2

        # Create the circular mask based on self.viewed_radius.
        circular_mask = (dist_sq <= self.viewed_radius**2).float()

        # Apply the circular mask to the input mask.
        masked_mask = mask * circular_mask
        return bool(masked_mask.sum() > 0)

    def run(self) -> None:
        with ThreadPoolExecutor() as executor:
            for frame_idx, results in enumerate(
                self.model.track(source=str(self.video_path), imgsz=640, stream=True)
            ):
                try:
                    gaze_position = self.get_gaze_position(frame_idx)
                    if gaze_position is None:
                        continue

                    boxes = []
                    rois = []
                    track_ids = []
                    for result in results:
                        mask = F.resize(
                            result.masks[0].data,
                            self.resolution,
                            interpolation=InterpolationMode.NEAREST,
                        ).squeeze()

                        if not self.mask_too_large(mask) and self.mask_was_viewed(
                            mask, gaze_position
                        ):
                            box = masks_to_boxes(mask.unsqueeze(0)).int().cpu().numpy()[0]
                            x1, y1, x2, y2 = box
                            roi = results[0].orig_img[y1:y2, x1:x2, :]
                            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                            boxes.append(box)
                            rois.append(roi)
                            track_ids.append(int(result.boxes.id[0]))

                    if len(boxes) > 0:
                        # Offload saving with thread pool (asynchronously)
                        rois_array = np.empty(len(rois), dtype=object)
                        for i, roi in enumerate(rois):
                            rois_array[i] = roi

                        executor.submit(
                            np.savez_compressed,
                            self.results_path / f"{frame_idx}.npz",
                            boxes=boxes,
                            rois=rois_array,
                            track_ids=track_ids,
                            frame_idx=frame_idx,
                            gaze_position=gaze_position,
                        )

                except Exception as e:
                    print(f"Error processing frame {frame_idx}: {e}")
                    traceback.print_exc()
