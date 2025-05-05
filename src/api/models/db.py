from pathlib import Path

from sqlalchemy import ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.api.db import Base
from src.config import RECORDINGS_PATH, TRACKING_RESULTS_PATH
from src.utils import generate_pleasant_color


class Recording(Base):
    __tablename__ = "recordings"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    visible_name: Mapped[str] = mapped_column(String)
    participant: Mapped[str] = mapped_column(String)
    created: Mapped[str] = mapped_column(String)
    duration: Mapped[str] = mapped_column(String)

    calibration_recordings: Mapped[list["CalibrationRecording"]] = relationship(
        "CalibrationRecording", back_populates="recording"
    )

    @property
    def video_path(self) -> Path:
        return RECORDINGS_PATH / f"{self.id}.mp4"

    @property
    def gaze_data_path(self) -> Path:
        return RECORDINGS_PATH / f"{self.id}.tsv"


class SimRoom(Base):
    __tablename__ = "simrooms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String)

    calibration_recordings: Mapped[list["CalibrationRecording"]] = relationship(
        "CalibrationRecording", back_populates="simroom", cascade="all, delete-orphan"
    )
    classes: Mapped[list["SimRoomClass"]] = relationship(
        "SimRoomClass", back_populates="simroom", cascade="all, delete-orphan"
    )


class SimRoomClass(Base):
    __tablename__ = "classes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    simroom_id: Mapped[int] = mapped_column(Integer, ForeignKey("simrooms.id"))
    class_name: Mapped[str] = mapped_column(String)
    color: Mapped[str] = mapped_column(String, default=generate_pleasant_color)

    simroom: Mapped["SimRoom"] = relationship("SimRoom", back_populates="classes")
    annotations: Mapped[list["Annotation"]] = relationship(
        "Annotation", back_populates="simroom_class", cascade="all, delete-orphan"
    )


class CalibrationRecording(Base):
    __tablename__ = "calibration_recordings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    simroom_id: Mapped[int] = mapped_column(Integer, ForeignKey("simrooms.id"))
    recording_id: Mapped[str] = mapped_column(String, ForeignKey("recordings.id"))

    simroom: Mapped["SimRoom"] = relationship(
        "SimRoom", back_populates="calibration_recordings"
    )
    recording: Mapped[Recording] = relationship(
        "Recording", back_populates="calibration_recordings"
    )
    annotations: Mapped[list["Annotation"]] = relationship(
        "Annotation", back_populates="calibration_recording", cascade="all, delete-orphan"
    )

    @property
    def video_path(self) -> Path:
        return RECORDINGS_PATH / f"{self.recording_id}.mp4"

    @property
    def tracking_results_path(self) -> Path:
        return TRACKING_RESULTS_PATH / str(self.id)

    @property
    def tracking_result_paths(self) -> list[Path]:
        return list(self.tracking_results_path.iterdir())


class Annotation(Base):
    __tablename__ = "annotations"
    __table_args__ = (
        UniqueConstraint(
            "calibration_id",
            "frame_idx",
            "simroom_class_id",
            name="_calibration_frame_class_uc",
            sqlite_on_conflict="ROLLBACK",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    calibration_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("calibration_recordings.id")
    )
    simroom_class_id: Mapped[int] = mapped_column(Integer, ForeignKey("classes.id"))
    frame_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    mask_base64: Mapped[str] = mapped_column(String, nullable=True)
    frame_crop_base64: Mapped[str] = mapped_column(String, nullable=True)
    box_json: Mapped[str] = mapped_column(String, nullable=True)

    calibration_recording: Mapped["CalibrationRecording"] = relationship(
        "CalibrationRecording", back_populates="annotations"
    )
    simroom_class: Mapped["SimRoomClass"] = relationship(
        "SimRoomClass", back_populates="annotations"
    )
    point_labels: Mapped[list["PointLabel"]] = relationship(
        "PointLabel", back_populates="annotation", cascade="all, delete-orphan"
    )


class PointLabel(Base):
    __tablename__ = "point_labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    annotation_id: Mapped[int] = mapped_column(Integer, ForeignKey("annotations.id"))
    x: Mapped[int] = mapped_column(Integer)
    y: Mapped[int] = mapped_column(Integer)
    label: Mapped[int] = mapped_column(Integer)

    annotation: Mapped["Annotation"] = relationship(
        "Annotation", back_populates="point_labels"
    )
