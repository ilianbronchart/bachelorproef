from typing import Any, Union, cast, no_type_check

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, Session, joinedload, mapped_column, relationship
from sqlalchemy_serializer import SerializerMixin

from src.db.db import Base, engine
from src.db.models.recording import Recording
from src.utils import generate_pleasant_color


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

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sim_room_id: Mapped[int] = mapped_column(Integer, ForeignKey("sim_rooms.id"))
    class_name: Mapped[str] = mapped_column(String)
    color: Mapped[str] = mapped_column(String, default=generate_pleasant_color)

    sim_room: Mapped["SimRoom"] = relationship("SimRoom", back_populates="classes")
    # Each class can be used in many annotations
    annotations: Mapped[list["Annotation"]] = relationship(
        "Annotation", back_populates="sim_room_class", cascade="all, delete-orphan"
    )

    @no_type_check
    def to_dict(self) -> dict[str, Any]:
        return cast(SerializerMixin, super()).to_dict(rules=["-annotations", "-sim_room"])


# CalibrationRecording entity (links a Recording with a SimRoom for calibration/grounding)
class CalibrationRecording(Base, SerializerMixin):
    __tablename__ = "calibration_recordings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sim_room_id: Mapped[int] = mapped_column(Integer, ForeignKey("sim_rooms.id"))
    recording_uuid: Mapped[str] = mapped_column(String, ForeignKey("recordings.uuid"))

    sim_room: Mapped["SimRoom"] = relationship("SimRoom", back_populates="calibration_recordings")
    recording: Mapped[Recording] = relationship("Recording", back_populates="calibration_recordings")
    annotations: Mapped[list["Annotation"]] = relationship("Annotation", back_populates="calibration_recording")

    @staticmethod
    def get_all() -> list["CalibrationRecording"]:
        with Session(engine) as session:
            return session.query(CalibrationRecording).options(joinedload(CalibrationRecording.recording)).all()

    @staticmethod
    def get(id: int | str) -> Union["CalibrationRecording", None]:
        with Session(engine) as session:
            return (
                session.query(CalibrationRecording)
                .options(joinedload(CalibrationRecording.recording))
                .filter(CalibrationRecording.id == id)
                .first()
            )


# Annotation entity (links a calibration recording with a specific SimRoomClass)
class Annotation(Base, SerializerMixin):
    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    calibration_recording_id: Mapped[int] = mapped_column(Integer, ForeignKey("calibration_recordings.id"))
    sim_room_class_id: Mapped[int] = mapped_column(Integer, ForeignKey("classes.id"))
    frame_time: Mapped[float] = mapped_column(Float, nullable=False)  # Time in seconds

    calibration_recording: Mapped["CalibrationRecording"] = relationship(
        "CalibrationRecording", back_populates="annotations"
    )
    sim_room_class: Mapped["SimRoomClass"] = relationship("SimRoomClass", back_populates="annotations")
    # Each annotation can have multiple point labels
    point_labels: Mapped[list["PointLabel"]] = relationship("PointLabel", back_populates="annotation")


# PointLabel entity (stores (x, y) coordinates and a binary label)
class PointLabel(Base, SerializerMixin):
    __tablename__ = "point_labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    annotation_id: Mapped[int] = mapped_column(Integer, ForeignKey("annotations.id"))
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)
    label: Mapped[bool] = mapped_column(Boolean)

    annotation: Mapped["Annotation"] = relationship("Annotation", back_populates="point_labels")
