import math
from pathlib import Path
from typing import Any, Union

from sqlalchemy import ForeignKey, Integer, String, UniqueConstraint, event
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship
from sqlalchemy_serializer import SerializerMixin

from src.api.db import Base, engine
from src.config import LABELING_ANNOTATIONS_DIR, LABELING_RESULTS_PATH, RECORDINGS_PATH
from src.utils import generate_pleasant_color


class Recording(Base):
    __tablename__ = "recordings"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    visible_name: Mapped[str] = mapped_column(String)
    participant: Mapped[str] = mapped_column(String)
    created: Mapped[str] = mapped_column(String)
    duration: Mapped[str] = mapped_column(String)
    folder_name: Mapped[str] = mapped_column(String)

    calibration_recordings: Mapped[list["CalibrationRecording"]] = relationship(
        "CalibrationRecording", back_populates="recording"
    )

    @property
    def video_path(self) -> Path:
        return RECORDINGS_PATH / f"{self.id}.mp4"

    @property
    def gaze_data_path(self) -> Path:
        return RECORDINGS_PATH / f"{self.id}.tsv"


class SimRoom(Base, SerializerMixin):
    __tablename__ = "sim_rooms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String)

    calibration_recordings: Mapped[list["CalibrationRecording"]] = relationship(
        "CalibrationRecording", back_populates="sim_room", cascade="all, delete-orphan"
    )
    classes: Mapped[list["SimRoomClass"]] = relationship(
        "SimRoomClass", back_populates="sim_room", cascade="all, delete-orphan"
    )


class SimRoomClass(Base, SerializerMixin):
    __tablename__ = "classes"
    serialize_rules = ["-annotations", "-sim_room"]

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sim_room_id: Mapped[int] = mapped_column(Integer, ForeignKey("sim_rooms.id"))
    class_name: Mapped[str] = mapped_column(String)
    color: Mapped[str] = mapped_column(String, default=generate_pleasant_color)

    sim_room: Mapped["SimRoom"] = relationship("SimRoom", back_populates="classes")
    annotations: Mapped[list["Annotation"]] = relationship(
        "Annotation", back_populates="sim_room_class", cascade="all, delete-orphan"
    )


class CalibrationRecording(Base, SerializerMixin):
    __tablename__ = "calibration_recordings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sim_room_id: Mapped[int] = mapped_column(Integer, ForeignKey("sim_rooms.id"))
    recording_id: Mapped[str] = mapped_column(String, ForeignKey("recordings.id"))

    sim_room: Mapped["SimRoom"] = relationship(
        "SimRoom", back_populates="calibration_recordings"
    )
    recording: Mapped[Recording] = relationship(
        "Recording", back_populates="calibration_recordings"
    )
    annotations: Mapped[list["Annotation"]] = relationship(
        "Annotation", back_populates="calibration_recording", cascade="all, delete-orphan"
    )

    @property
    def labeling_results_path(self) -> Path:
        return LABELING_RESULTS_PATH / str(self.id)

    @property
    def annotations_path(self) -> Path:
        return self.labeling_results_path / LABELING_ANNOTATIONS_DIR

    @property
    def tracking_result_paths(self) -> list[Path]:
        return [
            res
            for res in self.labeling_results_path.iterdir()
            if res.name != LABELING_ANNOTATIONS_DIR
        ]


class Annotation(Base, SerializerMixin):
    __tablename__ = "annotations"
    __table_args__ = (
        UniqueConstraint(
            "calibration_recording_id",
            "frame_idx",
            "sim_room_class_id",
            name="_calibration_frame_class_uc",
            sqlite_on_conflict="ROLLBACK",
        ),
    )
    serialize_rules = ["-point_labels", "-calibration_recording", "-sim_room_class"]

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    calibration_recording_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("calibration_recordings.id")
    )
    sim_room_class_id: Mapped[int] = mapped_column(Integer, ForeignKey("classes.id"))
    frame_idx: Mapped[int] = mapped_column(Integer, nullable=False)

    calibration_recording: Mapped["CalibrationRecording"] = relationship(
        "CalibrationRecording", back_populates="annotations"
    )
    sim_room_class: Mapped["SimRoomClass"] = relationship(
        "SimRoomClass", back_populates="annotations"
    )
    point_labels: Mapped[list["PointLabel"]] = relationship(
        "PointLabel", back_populates="annotation", cascade="all, delete-orphan"
    )

    @property
    def result_path(self) -> Path:
        return (
            LABELING_RESULTS_PATH
            / str(self.calibration_recording_id)
            / LABELING_ANNOTATIONS_DIR
            / f"{self.sim_room_class_id}_{self.frame_idx}.npz"
        )


# TODO: REMOVE THIS AFTER REFACTOR
@event.listens_for(Annotation, "after_delete")
def remove_annotation_result(_mapper: Any, _connection: Any, target: Annotation) -> None:
    if target.result_path.exists():
        target.result_path.unlink()


class PointLabel(Base, SerializerMixin):
    serialize_rules = ["-annotation"]
    __tablename__ = "point_labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    annotation_id: Mapped[int] = mapped_column(Integer, ForeignKey("annotations.id"))
    x: Mapped[int] = mapped_column(Integer)
    y: Mapped[int] = mapped_column(Integer)
    label: Mapped[int] = mapped_column(Integer)

    annotation: Mapped["Annotation"] = relationship(
        "Annotation", back_populates="point_labels"
    )

    @staticmethod
    def find_closest(
        annotation_id: int, x: int, y: int, max_distance: int = 1
    ) -> Union["PointLabel", None]:
        """
        Find the closest point label to the given coordinates within an annotation.

        Args:
            annotation_id: The ID of the annotation to search within
            x: X-coordinate of the reference point
            y: Y-coordinate of the reference point

        Returns:
            The closest PointLabel or None if there are no points for the annotation

        Raises:
            ValueError: If the closest point exceeds MAX_DISTANCE
        """

        with Session(engine) as session:
            points = (
                session.query(PointLabel)
                .filter(PointLabel.annotation_id == annotation_id)
                .all()
            )

            if not points:
                return None

            closest_point = None
            min_distance = float("inf")

            for point in points:
                distance = math.sqrt((point.x - x) ** 2 + (point.y - y) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_point = point

            # Check if the closest point exceeds MAX_DISTANCE
            if min_distance > max_distance:
                return None

            return closest_point
