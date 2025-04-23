import concurrent.futures
import os
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from src.config import GAZE_FOVEA_FOV, TOBII_FOV_X
from src.api.controllers.gaze_controller import (
    get_gaze_points,
    match_frames_to_gaze,
    parse_gazedata_file,
)
from src.utils import cv2_video_fps, cv2_video_resolution, extract_frames_to_dir
from torchvision.ops import masks_to_boxes
from tqdm import tqdm
from ultralytics import FastSAM
from ultralytics.engine.results import Results


class GazeSegmentationJob:
    def __init__(
        self,
        video_path: Path,
        gaze_data_path: Path,
        results_path: Path,
        batch_size: int = 50,
        crop_size: int = 512,
        fovea_fov: float = GAZE_FOVEA_FOV,
        fov_x: float = TOBII_FOV_X,
        checkpoint_path: str = "checkpoints/FastSAM-x.pt",
        frames_path: Path | None = None,
    ):
        self.video_path = video_path
        self.gaze_data_path = gaze_data_path
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.half_crop = crop_size // 2
        self.fovea_fov = fovea_fov
        self.fov_x = fov_x

        # Set up the frames directory.
        if frames_path is None:
            self.frames_path = Path(tempfile.mkdtemp())
            extract_frames_to_dir(video_path, self.frames_path)
        elif not frames_path.exists():
            raise ValueError(f"Frames directory does not exist: {frames_path}")
        else:
            self.frames_path = frames_path

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
        self.fps = cv2_video_fps(self.video_path)
        self.viewed_radius = int((self.fovea_fov / self.fov_x) * self.resolution[1])

        # Parse gaze data.
        self.gaze_data = parse_gazedata_file(self.gaze_data_path)
        self.gaze_points = get_gaze_points(self.gaze_data, self.resolution)

        # Get all frame file paths (assuming names are frame indices).
        self.frames = sorted(
            [
                frame
                for frame in self.frames_path.iterdir()
                if frame.suffix.lower() == ".jpg"
            ],
            key=lambda x: int(x.stem),
        )
        # Map frame indexes to gaze points.
        self.frame_gaze_mapping = match_frames_to_gaze(
            len(self.frames), self.gaze_points, self.fps
        )
        # Batch the frame file paths.
        self.frame_batches = [
            self.frames[i : i + self.batch_size]
            for i in range(0, len(self.frames), self.batch_size)
        ]

        # Timing statistics.
        self.total_batch_load_time = 0.0
        self.total_inference_time = 0.0
        self.total_postprocess_time = 0.0

    def get_gaze_position(self, frame_idx: int) -> tuple[int, int] | None:
        """
        Get the gaze position for a frame index.
        """
        gaze_points = self.frame_gaze_mapping[frame_idx]
        if len(gaze_points) == 0:
            return None
        return gaze_points[0].position

    def load_image(self, frame_path: Path, gaze_point: tuple[int, int]) -> torch.Tensor:
        """
        Load an image from disk, crop around the gaze point, and normalize.
        """
        try:
            if not frame_path.stem.isdigit():
                raise ValueError(
                    f"Frame name should be the frame index: {frame_path.stem}"
                )

            img = cv2.imread(str(frame_path))
            if img is None:
                raise ValueError(f"Failed to load image: {frame_path}")

            # Convert image to a CUDA tensor in CHW format.
            img = torch.from_numpy(img).to("cuda").permute(2, 0, 1)
            cx, cy = gaze_point
            img_crop = F.crop(
                img,
                cy - self.half_crop,
                cx - self.half_crop,
                self.crop_size,
                self.crop_size,
            )
            return img_crop.float() / 255.0
        except Exception as e:
            print(f"Error loading frame {frame_path}: {e}")

    def filter_large_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Filter out masks with area greater than 30% of the frame area

        Args:
            masks: tensor containing masks of shape (N, H, W)
        """
        if len(masks) == 0:
            return masks

        _, height, width = masks.shape
        frame_area = height * width
        max_mask_area = 0.3 * frame_area

        mask_areas = masks.sum(dim=(1, 2))
        filtered_masks = masks[mask_areas <= max_mask_area]
        return filtered_masks

    def filter_viewed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Filter out masks that are not within the viewed radius of the gaze point.
        Masks are the size of the crop around the gaze point, so the gaze point is at the center.
        """
        if len(masks) == 0:
            return masks

        sample_mask = masks[0]
        height, width = sample_mask.shape
        device = sample_mask.device

        # Create a circular mask centered at the gaze point.
        y = torch.arange(0, height, device=device).view(-1, 1).repeat(1, width)
        x = torch.arange(0, width, device=device).view(1, -1).repeat(height, 1)
        dist_sq = (x - self.half_crop) ** 2 + (y - self.half_crop) ** 2
        circular_mask = (
            (dist_sq <= self.viewed_radius**2).float().unsqueeze(0)
        )  # (1, H, W)

        # Apply the circular mask.
        masked_masks = masks * circular_mask
        mask_areas = masked_masks.sum(dim=(1, 2))
        return masks[mask_areas > 0]

    def postprocess_result(
        self, frame: torch.Tensor, frame_idx: int, masks: torch.Tensor, gaze_point: tuple[int, int]
    ) -> None:
        """
        This function is called concurrently to postprocess the results of a frame.
        The masks are filtered to remove large masks and masks outside the viewed radius, and then cropped to the bounding boxes.
        The bounding boxes are calculated from the masks, then adjusted to the coordinates of the original frame.
        Regions of interests are cropped for each bounding box.
        The results are saved to a .npz file.

        Args:
            frame: cropped frame tensor
            frame_idx: frame index
            masks: masks tensor (same shape as frame)
            gaze_point: gaze point (x, y)
        """
        try:
            filtered_masks = self.filter_large_masks(masks)
            viewed_masks = self.filter_viewed_masks(filtered_masks)

            boxes = masks_to_boxes(viewed_masks).int().cpu().numpy()

            # Crop masks to the bounding boxes.
            cropped_masks = np.empty(len(boxes), dtype=object)
            for i, mask in enumerate(viewed_masks):
                x1, y1, x2, y2 = boxes[i]
                mask_np = mask.detach().cpu().numpy()
                cropped_masks[i] = mask_np[y1:y2, x1:x2]

            # Extract regions of interest (ROI) for each bounding box.
            rois = np.empty(len(boxes), dtype=object)
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                cropped_frame = frame[:, y1:y2, x1:x2] * 255
                roi = cropped_frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                rois[i] = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            # Adjust bounding boxes to the original frame coordinates.
            cx, cy = gaze_point
            crop_left, crop_top = max(cx - self.half_crop, 0), max(cy - self.half_crop, 0)
            orig_frame_boxes = boxes + np.array([crop_left, crop_top, crop_left, crop_top])

            np.savez_compressed(
                self.results_path / f"{frame_idx}.npz",
                boxes=orig_frame_boxes,
                masks=cropped_masks,
                rois=rois,
            )
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")

    def process_batches(self) -> None:
        """
        For each batch of frames: load images, run inference, and postprocess results.
        """
        for batch in tqdm(self.frame_batches, desc="Batches"):
            # Load images concurrently.
            start_time = time.time()
            batch_frame_indexes = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for image in batch:
                    if not image.stem.isdigit():
                        raise ValueError(
                            f"Frame name should be the frame index: {image.stem}"
                        )

                    frame_idx = int(image.stem)
                    gaze_position = self.get_gaze_position(frame_idx)

                    if gaze_position is not None:
                        futures.append(
                            executor.submit(self.load_image, image, gaze_position)
                        )
                        batch_frame_indexes.append(frame_idx)

                batch_tensor = torch.stack([future.result() for future in futures])
            self.total_batch_load_time += time.time() - start_time

            # Run inference on the batch and measure GPU time.
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            results: list[Results] = self.model.predict(
                source=batch_tensor,
                retina_masks=True,
                device="cuda",
                verbose=False,
                imgsz=self.crop_size,
            )
            end_event.record()
            torch.cuda.synchronize()  # Ensure GPU operations have finished.
            self.total_inference_time += start_event.elapsed_time(end_event) / 1000.0

            # Postprocess results concurrently.
            start = time.time()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for i, result in enumerate(results):
                    if result.masks is None or len(result.masks) == 0:
                        continue
                    frame_idx = batch_frame_indexes[i]
                    frame = batch_tensor[i]
                    gaze_point = self.get_gaze_position(frame_idx)
                    futures.append(
                        executor.submit(
                            self.postprocess_result,
                            frame,
                            frame_idx,
                            result.masks.data,
                            gaze_point,
                        )
                    )
                concurrent.futures.wait(futures)
            self.total_postprocess_time += time.time() - start

        print(f"Total batch load time: {self.total_batch_load_time:.2f} seconds")
        print(f"Total inference time: {self.total_inference_time:.2f} seconds")
        print(f"Total postprocess time: {self.total_postprocess_time:.2f} seconds")

    def process_result_frame(
        self,
        original_frame: Path,
        frame_result: Path,
        result_frames_dir: Path,
        gaze_point: tuple[int, int] | None = None,
    ) -> None:
        """
        Overlay each mask as a red translucent region (alpha 0.3) and draw its bounding box
        on the original frame. The masks are drawn at the bounding box location since they
        were saved cropped to their bounding box. Also overlay the gaze point as a circle if provided.
        """
        frame = cv2.imread(str(original_frame))
        if frame is None:
            raise ValueError(f"Failed to load frame: {original_frame}")

        data = np.load(frame_result, allow_pickle=True)
        boxes = data["boxes"]  # Expected shape: [N, 4] with (x1, y1, x2, y2)
        masks = data["masks"]  # A list or array of cropped mask arrays.

        alpha = 0.3  # Opacity for the red overlay.
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            roi = frame[y1:y2, x1:x2]
            mask = masks[i]

            # Ensure the mask is binary (0 or 255).
            if mask.dtype != np.uint8:
                mask = (mask > 0.5).astype(np.uint8) * 255

            # Resize the mask to match ROI if necessary.
            if mask.shape != roi.shape[:2]:
                mask = cv2.resize(
                    mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST
                )

            red_overlay = np.full_like(roi, (0, 0, 255))  # Red in BGR.
            mask_bool = mask.astype(bool)
            roi[mask_bool] = cv2.addWeighted(
                roi[mask_bool], 1 - alpha, red_overlay[mask_bool], alpha, 0
            )

            # Draw the bounding box.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # If a gaze point is provided, draw a circle on the frame.
        if gaze_point is not None:
            cv2.circle(frame, gaze_point, self.viewed_radius, (255, 0, 0), 2)

        result_frames_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(result_frames_dir / f"{original_frame.stem}.jpg"), frame)

    def create_video_from_results(self, output_video: Path) -> None:
        """
        Process saved .npz result frames, overlay masks and gaze point on the original frames,
        and run ffmpeg to create a video.
        """
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            concurrent.futures.ThreadPoolExecutor() as executor,
        ):
            futures = []
            for batch in self.frame_batches:
                for frame in batch:
                    frame_idx = int(frame.stem)
                    frame_result = self.results_path / f"{frame_idx}.npz"
                    if frame_result.exists():
                        gaze_point = self.get_gaze_position(frame_idx)
                        futures.append(
                            executor.submit(
                                self.process_result_frame,
                                frame,
                                frame_result,
                                Path(tmpdir),
                                gaze_point,
                            )
                        )
            concurrent.futures.wait(futures)
            # Use ffmpeg to create a video (frames are read in glob order, so naming matters).
            cmd = f'ffmpeg -hwaccel cuda -y -pattern_type glob -framerate {self.fps} -i "{tmpdir}/*.jpg" -c:v libx264 -pix_fmt yuv420p "{output_video}"'
            os.system(cmd)
