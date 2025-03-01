import math
import shutil
from pathlib import Path
from typing import Any, Union, no_type_check

from sqlalchemy import ForeignKey, Integer, String, UniqueConstraint, event
from sqlalchemy.orm import Mapped, Session, joinedload, mapped_column, relationship
from sqlalchemy_serializer import SerializerMixin

from src.config import LABELING_ANNOTATIONS_DIR, LABELING_RESULTS_PATH
from src.db.db import Base, engine
from src.utils import generate_pleasant_color

from .recording import Recording


# SimRoom entity (defines a simulation room)
class SimRoom(Base, SerializerMixin):
    __tablename__ = "sim_rooms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String)
    color: Mapped[str] = mapped_column(String, default="#000000")  # TODO remove this

    # Add cascade="all, delete-orphan" to automatically delete related calibration recordings
    calibration_recordings: Mapped[list["CalibrationRecording"]] = relationship(
        "CalibrationRecording", back_populates="sim_room", cascade="all, delete-orphan"
    )
    classes: Mapped[list["SimRoomClass"]] = relationship(
        "SimRoomClass", back_populates="sim_room", cascade="all, delete-orphan"
    )


# SimRoomClass entity (defines labeling classes for a SimRoom)
class SimRoomClass(Base, SerializerMixin):
    __tablename__ = "classes"
    serialize_rules = ["-annotations", "-sim_room"]

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sim_room_id: Mapped[int] = mapped_column(Integer, ForeignKey("sim_rooms.id"))
    class_name: Mapped[str] = mapped_column(String)
    color: Mapped[str] = mapped_column(String, default=generate_pleasant_color)

    sim_room: Mapped["SimRoom"] = relationship("SimRoom", back_populates="classes")
    # Each class can be used in many annotations
    annotations: Mapped[list["Annotation"]] = relationship(
        "Annotation", back_populates="sim_room_class", cascade="all, delete-orphan"
    )


event.listens_for(SimRoomClass, "after_delete")
def remove_sim_room_class_annotations(
    _mapper: Any, _connection: Any, target: SimRoomClass
) -> None:
    with Session(engine) as session:
        # Remove all annotations for the class
        annotations = session.query(Annotation).filter(Annotation.sim_room_class_id == target.id)
        for annotation in annotations:
            if annotation.result_path.exists():
                shutil.rmtree(annotation.result_path)

        # Remove all tracking results for the class
        for labeling_results in LABELING_RESULTS_PATH.iterdir():
            tracking_results_path = labeling_results / target.id
            if tracking_results_path.exists():
                shutil.rmtree(tracking_results_path)


def remove_sim_room_class_labeling_results(
    _mapper: Any, _connection: Any, target: SimRoomClass
) -> None:
    with Session(engine) as session:
        calibration_recordings = session.query(CalibrationRecording).all()
        for calibration_recording in calibration_recordings:
            shutil.rmtree(calibration_recording.labeling_results_path / str(target.id))


# CalibrationRecording entity (links a Recording with a SimRoom for calibration/grounding)
class CalibrationRecording(Base, SerializerMixin):
    __tablename__ = "calibration_recordings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sim_room_id: Mapped[int] = mapped_column(Integer, ForeignKey("sim_rooms.id"))
    recording_uuid: Mapped[str] = mapped_column(String, ForeignKey("recordings.uuid"))

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

    @staticmethod
    def get_all() -> list["CalibrationRecording"]:
        with Session(engine) as session:
            return (
                session.query(CalibrationRecording)
                .options(joinedload(CalibrationRecording.recording))
                .all()
            )

    @staticmethod
    def get(id_: int | str) -> Union["CalibrationRecording", None]:
        with Session(engine) as session:
            return (
                session.query(CalibrationRecording)
                .options(joinedload(CalibrationRecording.recording))
                .filter(CalibrationRecording.id == id_)
                .first()
            )


# Event listeners to create and remove labeling results directory
@event.listens_for(CalibrationRecording, "after_insert")
def create_labeling_results_path(
    _mapper: Any, _connection: Any, target: CalibrationRecording
) -> None:
    target.labeling_results_path.mkdir(parents=True, exist_ok=True)
    target.annotations_path.mkdir(parents=True, exist_ok=True)


@event.listens_for(CalibrationRecording, "after_delete")
def remove_labeling_results_path(
    _mapper: Any, _connection: Any, target: CalibrationRecording
) -> None:
    if target.labeling_results_path.exists():
        shutil.rmtree(target.labeling_results_path)


# Annotation entity (links a calibration recording with a specific SimRoomClass)
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

    def __init__(
        self,
        calibration_recording_id: int,
        sim_room_class_id: int,
        frame_idx: int,
    ) -> None:
        self.calibration_recording_id = calibration_recording_id
        self.sim_room_class_id = sim_room_class_id
        self.frame_idx = frame_idx

    @no_type_check
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "calibration_recording_id": self.calibration_recording_id,
            "sim_room_class_id": self.sim_room_class_id,
            "frame_idx": self.frame_idx,
            "point_labels": [label.to_dict() for label in self.point_labels],
        }


@event.listens_for(Annotation, "after_delete")
def remove_annotation_result(_mapper: Any, _connection: Any, target: Annotation) -> None:
    if target.result_path.exists():
        target.result_path.unlink()


# PointLabel entity (stores (x, y) coordinates and a binary label)
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
