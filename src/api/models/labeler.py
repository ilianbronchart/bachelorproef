import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageColor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sqlalchemy.orm import Session

from src.aliases import UInt8Array
from src.api.jobs.labeler_tracking import TrackingJob
from src.api.models.context import LabelingContext, Request
from src.config import (
    LABELING_ANNOTATIONS_DIR,
    RECORDINGS_PATH,
    Sam2Checkpoints,
)
from src.api.db import engine
from src.api.models.db.calibration import Annotation, CalibrationRecording, SimRoomClass
from src.api.controllers.sam2_controller import (
    load_sam2_predictor,
    predict_sam2,
)
from src.utils import extract_frames_to_dir, get_frame_from_dir


class Labeler:
    tracking_job: TrackingJob | None = None
    selected_class_id: int = -1
    show_inactive_classes: bool = True

    def __init__(self, calibration_recording: CalibrationRecording):
        self.calibration_recording: CalibrationRecording = calibration_recording
        self.video_path: Path = RECORDINGS_PATH / (
            calibration_recording.recording_uuid + ".mp4"
        )

        self.frames_path: Path = Path(tempfile.mkdtemp())
        extract_frames_to_dir(video_path=self.video_path, frames_path=self.frames_path)
        self.frame_count = len(list(self.frames_path.glob("*.jpg")))
        self.current_frame = self.get_frame(0)
        self.current_frame_idx = 0

        self.image_predictor: SAM2ImagePredictor = load_sam2_predictor(
            Sam2Checkpoints.LARGE
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
        return get_frame_from_dir(frame_idx, self.frames_path)

    def get_labeling_context(self, request: Request) -> LabelingContext:
        return LabelingContext(
            request=request,
            sim_room_id=self.calibration_recording.sim_room_id,
            recording_uuid=self.calibration_recording.recording_uuid,
            show_inactive_classes=self.show_inactive_classes,
        )

    def seek(self, frame_idx: int) -> None:
        if self.current_frame_idx == frame_idx:
            return

        self.current_frame_idx = frame_idx
        self.current_frame = self.get_frame(frame_idx)
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
                    Annotation.frame_idx == self.current_frame_idx,
                )
                .all()
            )

            annotation_class_ids = [
                annotation.sim_room_class_id for annotation in annotations
            ]

            sim_room_classes = (
                session.query(SimRoomClass)
                .filter(
                    SimRoomClass.id.in_(tracked_class_ids),
                    ~SimRoomClass.id.in_(annotation_class_ids),
                )
                .all()
            )

            results: list[tuple[SimRoomClass, UInt8Array, UInt8Array]] = []
            for ann in annotations:
                if (
                    not self.show_inactive_classes
                    and ann.sim_room_class_id != self.selected_class_id
                ):
                    continue

                file = np.load(ann.result_path)
                results.append((ann.sim_room_class, file["mask"], file["box"]))

            for sim_room_class in sim_room_classes:
                if (
                    not self.show_inactive_classes
                    and sim_room_class.id != self.selected_class_id
                ):
                    continue

                class_id = sim_room_class.id
                class_path = labeling_results_path / str(class_id)

                for result in class_path.glob("*.npz"):
                    frame_idx = int(result.stem)
                    if frame_idx != self.current_frame_idx:
                        continue

                    file = np.load(result)
                    results.append((sim_room_class, file["mask"], file["box"]))

        # Draw masks first
        for _, mask, box in results:
            self.draw_mask(frame, mask, box)

        # Draw bounding boxes
        for sim_room_class, _, box in results:
            self.draw_box(frame, box, sim_room_class)

        return frame

    def draw_mask(
        self, frame: UInt8Array, mask: UInt8Array, box: UInt8Array
    ) -> UInt8Array:
        x1, y1, x2, y2 = box

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

    def draw_box(
        self, frame: UInt8Array, box: UInt8Array, sim_room_class: SimRoomClass
    ) -> UInt8Array:
        x1, y1, x2, y2 = box
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

    def predict_image(
        self, annotation: Annotation, points: list[tuple[int, int]], labels: list[int]
    ) -> None:
        mask, box = predict_sam2(
            predictor=self.image_predictor,
            points=points,
            points_labels=labels,
        )

        if mask is None:
            raise ValueError("Failed to predict mask")

        np.savez_compressed(annotation.result_path, mask=mask, box=box)

    def create_tracking_job(self, annotations: list[Annotation]) -> None:
        if self.tracking_job is not None:
            raise ValueError("A tracking job is already in progress")

        self.tracking_job = TrackingJob(
            annotations=annotations,
            frames_path=self.frames_path,
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
