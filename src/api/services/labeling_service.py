import shutil
import tempfile
import threading
from collections.abc import Generator
from pathlib import Path

import numpy as np
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sqlalchemy.orm import Session
from torchvision.ops import masks_to_boxes

from src.aliases import UInt8Array
from src.api.models.pydantic import AnnotationDTO, CalibrationRecordingDTO
from src.api.repositories import annotations_repo, simrooms_repo
from src.api.services import sam2_service, simrooms_service
from src.api.utils import image_utils
from src.config import (
    Sam2Checkpoints,
)
from src.utils import extract_frames_to_dir, get_frame_from_dir


class TrackingJob:
    GRACE_PERIOD: int = 25  # Number of frames to wait before considering a tracking loss
    progress: float = 0.0

    def __init__(
        self,
        annotations: list[AnnotationDTO],
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
        self.video_predictor = sam2_service.load_video_predictor(Sam2Checkpoints.LARGE)
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
                obj_id=annotation.simroom_class_id,
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


class Labeler:
    _tracking_job: TrackingJob | None = None
    _selected_class_id: int = -1
    _show_inactive_classes: bool = True

    def __init__(self, cal_rec: CalibrationRecordingDTO):
        self._cal_rec = cal_rec
        self._frames_path: Path = Path(tempfile.mkdtemp())

        extract_frames_to_dir(
            video_path=self._cal_rec.video_path, frames_path=self._frames_path
        )

        self._frame_count = len(list(self._frames_path.glob("*.jpg")))
        self._current_frame_idx = 0
        self._image_predictor: SAM2ImagePredictor = sam2_service.load_predictor(
            Sam2Checkpoints.LARGE
        )
        self._current_frame = get_frame_from_dir(
            self._current_frame_idx, self._frames_path
        )

    @property
    def results_path(self) -> Path:
        return self._cal_rec.labeling_results_path

    @property
    def current_class_results_path(self) -> Path:
        return self._cal_rec.labeling_results_path / str(self.selected_class_id)

    @property
    def simroom_id(self) -> int:
        return self._cal_rec.simroom_id

    @property
    def recording_id(self) -> int:
        return self._cal_rec.recording_id

    @property
    def calibration_id(self) -> int:
        return self._cal_rec.id

    @property
    def current_frame_idx(self) -> int:
        return self._current_frame_idx

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def frames_path(self) -> Path:
        return self._frames_path

    @property
    def show_inactive_classes(self) -> bool:
        return self._show_inactive_classes

    @property
    def selected_class_id(self) -> int:
        return self._selected_class_id

    @property
    def has_selected_class(self) -> bool:
        return self._selected_class_id != -1

    @property
    def image_predictor(self) -> SAM2ImagePredictor:
        return self._image_predictor

    @property
    def tracking_progress(self) -> int | None:
        if self._tracking_job is None:
            return None
        return self._tracking_job.progress

    @property
    def is_tracking_current_class(self) -> bool:
        return (
            self._tracking_job is not None
            and self._tracking_job.class_id == self._selected_class_id
        )

    @property
    def is_tracking(self) -> bool:
        return self._tracking_job is not None

    def seek(self, frame_idx: int) -> None:
        if self.current_frame_idx == frame_idx:
            return

        self._current_frame_idx = frame_idx
        self._current_frame = get_frame_from_dir(frame_idx, self._frames_path)
        self._image_predictor.set_image(self._current_frame)

    def set_selected_class_id(self, db: Session, class_id: int | None = None) -> None:
        if class_id is None:
            self._selected_class_id = -1
        else:
            simrooms_repo.get_simroom_class(  # check if class exists
                db=db, class_id=class_id
            )
            self._selected_class_id = class_id

    def set_show_inactive_classes(self, show: bool) -> None:
        self._show_inactive_classes = show

    def _get_overlay_draw_data(
        self, db: Session
    ) -> list[tuple[str, str, UInt8Array, UInt8Array]]:
        # Get all the annotations for the current frame
        current_frame_annotations = annotations_repo.get_annotations_by_frame_idx(
            db=db,
            calibration_id=self.calibration_id,
            frame_idx=self.current_frame_idx,
        )
        annotation_class_ids = {ann.simroom_class_id for ann in current_frame_annotations}

        # Get all tracked classes and filter out the ones which have an annotation in the current frame
        tracked_classes = simrooms_repo.get_tracked_classes(
            db=db,
            calibration_id=self.calibration_id,
        )
        tracked_classes = [
            cls_ for cls_ in tracked_classes if cls_.id not in annotation_class_ids
        ]

        # Filter out annotations and classes that are not active if show_inactive_classes is False
        if not self.show_inactive_classes:
            active_id = self.selected_class_id
            current_frame_annotations = [
                ann
                for ann in current_frame_annotations
                if ann.simroom_class_id != active_id
            ]
            tracked_classes = [cls_ for cls_ in tracked_classes if cls_.id != active_id]

        # Information needed to draw the overlay (class name, color, mask, box)
        results_to_draw: list[tuple[str, str, UInt8Array, UInt8Array]] = []

        # Add current frame annotations
        for ann in current_frame_annotations:
            file = np.load(ann.result_path)
            results_to_draw.append((
                ann.simroom_class.class_name,
                ann.simroom_class.color,
                file["mask"],
                file["box"],
            ))

        # Add tracking results for current frame
        for cls_ in tracked_classes:
            class_id = cls_.id
            class_path = self.results_path / str(class_id)
            result_path = class_path / f"{self.current_frame_idx}.npz"
            if not result_path.exists():
                continue

            file = np.load(result_path)
            results_to_draw.append((
                cls_.class_name,
                cls_.color,
                file["mask"],
                file["box"],
            ))

        return results_to_draw

    def get_current_frame_overlay(self, db: Session) -> UInt8Array:
        results_to_draw = self._get_overlay_draw_data(db)

        # Draw masks first to avoid overlapping
        frame = self._current_frame.copy()
        for _, _, mask, box in results_to_draw:
            image_utils.draw_mask(frame, mask, box)

        # Draw bounding boxes
        for class_name, color, _, box in results_to_draw:
            image_utils.draw_box(frame, box, class_name, color)

        return frame

    def start_tracking(self, annotations: list[AnnotationDTO]) -> None:
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


labeler = None


def load(
    db: Session,
    calibration_id: int,
) -> Labeler:
    global labeler
    cal_rec = simrooms_service.get_calibration_recording(
        db=db,
        calibration_id=calibration_id,
    )
    if labeler is None:
        labeler = Labeler(cal_rec)
    return labeler


def unload() -> None:
    global labeler
    pass  # TODO: Implement unload labeler
