import shutil
import threading
from collections.abc import Generator
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import ImageColor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sqlalchemy.orm import Session
from src.aliases import UInt8Array
from src.api.models.context import LabelingContext, Request
from src.config import (
    FRAMES_PATH,
    LABELING_ANNOTATIONS_DIR,
    RECORDINGS_PATH,
    Sam2Checkpoints,
)
from src.db import engine
from src.db.models.calibration import Annotation, CalibrationRecording, SimRoomClass
from src.logic.inference.sam_2 import (
    load_sam2_predictor,
    load_sam2_video_predictor,
    predict_sam2,
)
from src.utils import get_frame_from_dir
from torchvision.ops import masks_to_boxes


class TrackingJob:
    GRACE_PERIOD: int = 25  # Number of frames to wait before considering a tracking loss
    progress: float = 0.0

    def __init__(
        self,
        annotations: list[Annotation],
        results_path: Path,
        frame_count: int,
        class_id: int,
    ):
        self.annotations = sorted(annotations, key=lambda x: x.frame_idx)
        self.results_path = results_path
        self.frame_count = frame_count
        self.class_id = class_id

    def initialize(self):
        # Load the video predictor and initialize the inference state
        self.video_predictor = load_sam2_video_predictor(Sam2Checkpoints.LARGE)
        self.inference_state = self.video_predictor.init_state(
            video_path=str(FRAMES_PATH), async_loading_frames=True
        )  # type: ignore[no-untyped-call]

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

                    file_path = self.results_path / f"{out_frame_idx}.npz"
                    np.savez_compressed(
                        file_path,
                        mask=final_mask,
                        bbox=np.array([x1, y1, x2, y2]).astype(np.int32),
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


class Labeler:
    current_frame: UInt8Array = get_frame_from_dir(0, FRAMES_PATH)
    frame_count: int = len(list(FRAMES_PATH.glob("*.jpg")))
    image_predictor: SAM2ImagePredictor = load_sam2_predictor(Sam2Checkpoints.LARGE)
    tracking_job: TrackingJob | None = None
    current_frame_idx: int = -1
    selected_class_id: int = -1

    def __init__(self, calibration_recording: CalibrationRecording):
        self.calibration_recording: CalibrationRecording = calibration_recording
        self.video_path: Path = RECORDINGS_PATH / (
            calibration_recording.recording_uuid + ".mp4"
        )

    @property
    def results_path(self) -> Path:
        return self.calibration_recording.labeling_results_path

    @property
    def current_class_results_path(self) -> Path:
        return self.calibration_recording.labeling_results_path / str(
            self.selected_class_id
        )

    def get_frame(self, frame_idx: int) -> UInt8Array:
        return get_frame_from_dir(frame_idx, FRAMES_PATH)

    def get_labeling_context(self, request: Request) -> LabelingContext:
        return LabelingContext(
            request=request,
            sim_room_id=self.calibration_recording.sim_room_id,
            recording_uuid=self.calibration_recording.recording_uuid,
        )

    def seek(self, frame_idx: int) -> None:
        if self.current_frame_idx == frame_idx:
            return

        self.current_frame_idx = frame_idx
        self.current_frame = get_frame_from_dir(frame_idx, FRAMES_PATH)
        self.image_predictor.set_image(self.current_frame)

    def get_overlay(self) -> np.ndarray:
        frame = self.current_frame.copy()

        labeling_results_path = self.calibration_recording.labeling_results_path

        tracked_class_ids = [
            int(x.name)
            for x in labeling_results_path.iterdir()
            if x.is_dir() and x.name != LABELING_ANNOTATIONS_DIR
        ]

        with Session(engine) as session:
            annotations = (
                session.query(Annotation)
                .filter(
                    Annotation.calibration_recording_id == self.calibration_recording.id,
                    Annotation.frame_idx == self.current_frame_idx
                    )
                .all()
            )
            annotation_class_ids = [annotation.sim_room_class_id for annotation in annotations]

            sim_room_classes = (
                session.query(SimRoomClass).filter(
                    SimRoomClass.id.in_(tracked_class_ids),
                    ~SimRoomClass.id.in_(annotation_class_ids),
                ).all()
            )

            results: list[tuple[SimRoomClass, UInt8Array, UInt8Array]] = []
            for ann in annotations:
                file = np.load(ann.result_path)
                results.append((ann.sim_room_class, file["mask"], file["bbox"]))

            for sim_room_class in sim_room_classes:
                class_id = sim_room_class.id
                class_path = labeling_results_path / str(class_id)

                for result in class_path.glob("*.npz"):
                    frame_idx = int(result.stem)
                    if frame_idx != self.current_frame_idx:
                        continue

                    file = np.load(result)
                    results.append((sim_room_class, file["mask"], file["bbox"]))

        # Draw masks first
        for _, mask, bbox in results:
            self.draw_mask(frame, mask, bbox)

        # Draw bounding boxes
        for sim_room_class, _, bbox in results:
            self.draw_bbox(frame, bbox, sim_room_class)

        return frame

    def draw_mask(
        self, frame: UInt8Array, mask: UInt8Array, bbox: UInt8Array
    ) -> UInt8Array:
        x1, y1, x2, y2 = bbox

        roi = frame[y1:y2, x1:x2].copy()
        white_overlay = np.full(roi.shape, 255, dtype=np.uint8)
        # Squeeze the mask to remove the extra dimension
        binary_mask = mask.squeeze(0) > 0

        # Apply alpha blending (0.2) on the white overlay only at locations where the mask is present
        roi[binary_mask] = cv2.addWeighted(
            roi[binary_mask], 1, white_overlay[binary_mask], 0.2, 0
        )
        frame[y1:y2, x1:x2] = roi
        return frame

    def draw_bbox(
        self, frame: UInt8Array, bbox: UInt8Array, sim_room_class: SimRoomClass
    ) -> UInt8Array:
        x1, y1, x2, y2 = bbox
        color_rgb = ImageColor.getcolor(sim_room_class.color, "RGB")
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])  # type: ignore[index]
        label = sim_room_class.class_name

        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 10, y1),
            color_bgr,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2,
        )

        return frame

    def predict_image(self, annotation: Annotation, points: list[tuple[int, int]], labels: list[int]) -> None:
        mask, bbox = predict_sam2(
            predictor=self.image_predictor,
            points=points,
            points_labels=labels,
        )

        if mask is None:
            raise ValueError("Failed to predict mask")

        np.savez_compressed(annotation.result_path, mask=mask, bbox=bbox)

    def create_tracking_job(self, annotations: list[Annotation]) -> None:
        if self.tracking_job is not None:
            raise ValueError("A tracking job is already in progress")

        self.tracking_job = TrackingJob(
            annotations=annotations,
            results_path=self.current_class_results_path,
            frame_count=self.frame_count,
            class_id=self.selected_class_id,
        )

        def job_runner():
            self.tracking_job.run()
            self.tracking_job = None

        threading.Thread(target=job_runner).start()

    def get_tracks(self) -> list[tuple[int, int]]:
        if not self.current_class_results_path.exists():
            return []

        # Gather and sort the frame indices from the filenames
        results = list(self.current_class_results_path.glob("*.npz"))
        results_frame_idx = sorted(int(result.stem) for result in results)

        if not results_frame_idx:
            return []

        tracks = []
        start = results_frame_idx[0]
        prev = start

        for frame in results_frame_idx[1:]:
            # Check if the current frame is consecutive to the previous one
            if frame == prev + 1:
                prev = frame  # Continue the current track
            else:
                # End the current track and start a new one
                tracks.append((start, prev))
                start = frame
                prev = frame

        # Add the final track
        tracks.append((start, prev))
        return tracks
