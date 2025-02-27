import base64
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageColor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from src.aliases import UInt8Array
from src.api.models.context import LabelingContext, Request
from src.config import FRAMES_PATH, RECORDINGS_PATH, Sam2Checkpoints
from src.db.models.calibration import Annotation, CalibrationRecording
from src.logic.inference.sam_2 import (
    load_sam2_predictor,
    load_sam2_video_predictor,
    predict_sam2,
)
from src.utils import get_frame_from_dir


class TrackingJob:
    pass


@dataclass
class ImagePredictionResult:
    mask: UInt8Array
    bounding_box: tuple[int, int, int, int]
    frame_crop: UInt8Array


class Labeler:
    calibration_recording: CalibrationRecording
    video_path: Path
    current_frame: UInt8Array = get_frame_from_dir(0, FRAMES_PATH)
    frame_count: int = len(list(FRAMES_PATH.glob("*.jpg")))
    image_predictor: SAM2ImagePredictor = load_sam2_predictor(Sam2Checkpoints.LARGE)
    video_predictor: SAM2VideoPredictor = load_sam2_video_predictor(Sam2Checkpoints.LARGE)
    tracking_job: TrackingJob | None = None
    current_frame_idx: int = -1
    selected_class_id: int = -1

    def __init__(self, calibration_recording: CalibrationRecording):
        self.calibration_recording = calibration_recording
        self.video_path = RECORDINGS_PATH / (
            calibration_recording.recording_uuid + ".mp4"
        )
        self.seek(0)

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

    def get_overlay(self, annotations: list[Annotation]) -> np.ndarray:
        frame = self.current_frame.copy()
        frame = self.draw_masks(frame, annotations)
        frame = self.draw_bboxes(frame, annotations)
        return frame

    def draw_masks(self, frame: np.ndarray, annotations: list[Annotation]) -> UInt8Array:
        for annotation in annotations:
            x1, y1, x2, y2 = annotation.bbox
            mask_bytes = base64.b64decode(annotation.annotation_mask)
            mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
            mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)

            roi = frame[y1:y2, x1:x2].copy()
            white_overlay = np.full(roi.shape, 255, dtype=np.uint8)
            binary_mask = mask > 0

            # Apply alpha blending (0.2) on the white overlay only at locations where the mask is present
            roi[binary_mask] = cv2.addWeighted(
                roi[binary_mask], 1, white_overlay[binary_mask], 0.2, 0
            )
            frame[y1:y2, x1:x2] = roi

        return frame

    def draw_bboxes(self, frame: UInt8Array, annotations: list[Annotation]) -> UInt8Array:
        for annotation in annotations:
            x1, y1, x2, y2 = annotation.bbox
            color_rgb = ImageColor.getcolor(annotation.sim_room_class.color, "RGB")
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])  # type: ignore[index]
            label = annotation.sim_room_class.class_name

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
        self, points: list[tuple[int, int]], labels: list[int]
    ) -> ImagePredictionResult | None:
        mask, bounding_box = predict_sam2(
            predictor=self.image_predictor,
            points=points,
            points_labels=labels,
        )
        if mask is None or bounding_box is None:
            return None

        mask_rgb = (mask.repeat(3, axis=0).transpose(1, 2, 0) * 255).astype(np.uint8)
        x1, y1, x2, y2 = bounding_box
        frame_crop = self.current_frame[y1:y2, x1:x2]

        return ImagePredictionResult(
            mask=mask_rgb, bounding_box=bounding_box, frame_crop=frame_crop
        )
