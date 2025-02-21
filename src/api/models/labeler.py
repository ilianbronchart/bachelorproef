


from dataclasses import dataclass
from pathlib import Path
from sam2.sam2_image_predictor import SAM2ImagePredictor
from src.db.models.calibration import CalibrationRecording
from src.logic.inference.sam_2 import load_sam2_predictor
from src.utils import get_frame_from_dir
from src.config import FRAMES_PATH, Sam2Checkpoints, RECORDINGS_PATH

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

    def seek(self, percent: float) -> np.ndarray:
        frame_idx = int(self.frame_count * percent)
        self.current_frame = get_frame_from_dir(frame_idx, FRAMES_PATH)
        self.predictor.set_image(self.current_frame)
        
        return self.current_frame