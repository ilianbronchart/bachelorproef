from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import os

import cv2
from fastapi import Request as FastAPIRequest
from ultralytics import FastSAM
from src.config import FRAMES_PATH, RECORDINGS_PATH, Template
from src.db import Recording
from src.db.models.calibration import Annotation, SimRoom, CalibrationRecording, SimRoomClass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import App

class Request(FastAPIRequest):
    _app: "App"

    @property
    def app(self) -> "App":
        return self._app


@dataclass
class BaseContext:
    request: Request

    def to_dict(self):
        dict_ = {k: v for k, v in self.__dict__.items() if k != "request"}
        dict_["request"] = self.request
        return dict_


@dataclass
class GlassesConnectionContext(BaseContext):
    glasses_connected: bool
    battery_level: float


@dataclass
class RecordingsContext(BaseContext):
    recordings: list[Recording] = field(default_factory=list)
    glasses_connected: bool = False
    failed_connection: bool = False
    content: str = Template.RECORDINGS

@dataclass
class SimRoomsContext(BaseContext):
    recordings: list[Recording] = field(default_factory=list)
    sim_rooms: list[SimRoom] = field(default_factory=list)
    selected_sim_room: SimRoom | None = None
    calibration_recordings: list[CalibrationRecording] = field(default_factory=list)
    classes: list[SimRoomClass] = field(default_factory=list)
    content = Template.SIMROOMS

@dataclass
class LabelingContext(BaseContext):
    calibration_recording: CalibrationRecording
    sim_room: SimRoom
    recording: Recording
    model: FastSAM
    classes: list[SimRoomClass]
    annotations: list[Annotation]
    content: str = Template.LABELER

    def __post_init__(self):
        self.extract_frames()

        self.calibration_recording

    def extract_frames(self) -> str:
        for file in os.listdir(FRAMES_PATH):
            os.remove(os.path.join(FRAMES_PATH, file))

        video_path = RECORDINGS_PATH / self.recording.uuid + ".mp4"

        # Open the video file
        cap = cv2.VideoCapture(str(FRAMES_PATH))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {FRAMES_PATH}")

        frame_index = 0
        # Adjust the number of workers based on your disk I/O capacity and CPU cores
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
            futures: list[Future[bool]] = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Build a filename for this frame
                frame_filename = os.path.join(FRAMES_PATH, f"frame_{frame_index:06d}.png")
                # Offload the disk write to a thread
                futures.append(executor.submit(cv2.imwrite, frame_filename, frame))
                frame_index += 1

            # Wait for all write operations to finish
            for future in futures:
                future.result()

        cap.release()
