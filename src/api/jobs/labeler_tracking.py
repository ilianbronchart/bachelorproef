import shutil
from collections.abc import Generator
from pathlib import Path

import numpy as np
import torch
from src.aliases import UInt8Array
from src.config import (
    Sam2Checkpoints,
)
from src.db.models.calibration import Annotation
from src.api.controllers.sam2_controller import (
    load_sam2_video_predictor,
)
from torchvision.ops import masks_to_boxes


class TrackingJob:
    GRACE_PERIOD: int = 25  # Number of frames to wait before considering a tracking loss
    progress: float = 0.0

    def __init__(
        self,
        annotations: list[Annotation],
        frames_path: Path,
        results_path: Path,
        frame_count: int,
        class_id: int,
    ):
        self.annotations = sorted(annotations, key=lambda x: x.frame_idx)
        self.frames_path = frames_path
        self.results_path = results_path
        self.frame_count = frame_count
        self.class_id = class_id

    def initialize(self):
        # Load the video predictor and initialize the inference state
        self.video_predictor = load_sam2_video_predictor(Sam2Checkpoints.LARGE)
        self.inference_state = self.video_predictor.init_state(
            video_path=str(self.frames_path), async_loading_frames=True
        )  # type: ignore[no-untyped-call]
        self.img_std = self.inference_state["images"].img_std.cuda()
        self.img_mean = self.inference_state["images"].img_mean.cuda()

        # Remove the results directory if it already exists
        if self.results_path.exists():
            shutil.rmtree(self.results_path)
        self.results_path.mkdir(parents=True)

        # Add the initial points to the video predictor
        for annotation in self.annotations:
            point_labels = annotation.point_labels
            points = [
                (int(point_label.x), int(point_label.y)) for point_label in point_labels
            ]
            labels = [point_label.label for point_label in point_labels]

            self.video_predictor.add_new_points(  # type: ignore[no-untyped-call]
                inference_state=self.inference_state,
                frame_idx=annotation.frame_idx,
                obj_id=annotation.sim_room_class_id,
                points=points,
                labels=labels,
            )

    def track_until_loss(
        self, start_frame_idx: int, reverse: bool = False
    ) -> Generator[int, None, None]:
        tracking_loss = 0
        with torch.amp.autocast("cuda"):
            for (
                out_frame_idx,
                _,
                out_mask_logits,
            ) in self.video_predictor.propagate_in_video(
                inference_state=self.inference_state,
                start_frame_idx=start_frame_idx,
                reverse=reverse,
            ):
                yield out_frame_idx

                mask_torch = out_mask_logits[0] > 0.5
                if mask_torch.any():
                    tracking_loss = 0

                    # calculate bounding box and final mask
                    x1, y1, x2, y2 = (
                        masks_to_boxes(mask_torch)[0].cpu().numpy().astype(np.int32)
                    )
                    mask = mask_torch.cpu().numpy().astype(np.uint8)

                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    final_mask = mask[:, y1:y2, x1:x2]

                    frame: UInt8Array = self.inference_state["images"].last_loaded_image
                    frame_roi = frame[y1:y2, x1:x2, :]

                    file_path = self.results_path / f"{out_frame_idx}.npz"
                    np.savez_compressed(
                        file_path,
                        mask=final_mask,
                        box=np.array([x1, y1, x2, y2]).astype(np.int32),
                        roi=frame_roi,
                        class_id=self.class_id,
                        frame_idx=out_frame_idx,
                    )
                else:
                    tracking_loss += 1

                if tracking_loss >= self.GRACE_PERIOD:
                    break

    def run(self):
        self.initialize()

        annotations_frame_idx = [annotation.frame_idx for annotation in self.annotations]
        last_tracked_annotation = -1

        while last_tracked_annotation != len(self.annotations) - 1:
            start_frame_idx = annotations_frame_idx[last_tracked_annotation + 1]
            self.progress = start_frame_idx / self.frame_count

            # Track backwards until tracking loss:
            list(self.track_until_loss(start_frame_idx, reverse=True))

            # Track forwards until tracking loss:
            for frame_idx in self.track_until_loss(start_frame_idx):
                self.progress = frame_idx / self.frame_count
                if frame_idx in annotations_frame_idx:
                    last_tracked_annotation = annotations_frame_idx.index(frame_idx)
