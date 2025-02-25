from dataclasses import dataclass
from pathlib import Path
from PIL import ImageColor
import base64

import cv2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from src.db.models.calibration import CalibrationRecording, Annotation
from src.logic.inference.sam_2 import load_sam2_predictor
from src.utils import get_frame_from_dir
from src.config import FRAMES_PATH, Sam2Checkpoints, RECORDINGS_PATH, LABELING_RESULTS_PATH
from src.db import engine
from sqlalchemy.orm import Session

import numpy as np

class Labeler():
    video_path: Path
    current_frame: np.ndarray
    frame_count: int
    predictor: SAM2ImagePredictor

    def __init__(self, calibration_recording: CalibrationRecording):
        self.calibration_recording = calibration_recording
        self.video_path = RECORDINGS_PATH / (calibration_recording.recording_uuid + ".mp4")
        self.current_frame = get_frame_from_dir(0, FRAMES_PATH)
        self.frame_count = len(list(FRAMES_PATH.glob("*.jpg")))
        self.predictor = load_sam2_predictor(Sam2Checkpoints.LARGE)
        self.seek(0)

    def seek(self, frame_idx: int) -> None:
        self.current_frame = get_frame_from_dir(frame_idx, FRAMES_PATH)
        self.predictor.set_image(self.current_frame)

    def get_overlay(self, annotations: list[Annotation]) -> np.ndarray:
        frame = self.current_frame.copy()
        frame = self.draw_masks(frame, annotations)
        frame = self.draw_bboxes(frame, annotations)
        return frame
        
    def draw_masks(self, frame: np.ndarray, annotations: list[Annotation]) -> np.ndarray:
        for annotation in annotations:
            x1, y1, x2, y2 = annotation.bbox            
            mask_bytes = base64.b64decode(annotation.annotation_mask)
            mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
            mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)

            roi = frame[y1:y2, x1:x2].copy()
            white_overlay = np.full(roi.shape, 255, dtype=np.uint8)
            binary_mask = mask > 0

            # Apply alpha blending (0.2) on the white overlay only at locations where the mask is present
            roi[binary_mask] = cv2.addWeighted(roi[binary_mask], 1, white_overlay[binary_mask], 0.2, 0)
            frame[y1:y2, x1:x2] = roi
        
        return frame
    
    def draw_bboxes(self, frame: np.ndarray, annotations: list[Annotation]) -> np.ndarray:
        for annotation in annotations:
            x1, y1, x2, y2 = annotation.bbox
            color_rgb = ImageColor.getcolor(annotation.sim_room_class.color, "RGB")
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0]) # type: ignore
            label = annotation.sim_room_class.class_name

            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color_bgr, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)    

        return frame