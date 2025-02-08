from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship, Session, joinedload

from src.db.db import Base
import base64
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Union

from g3pylib.recordings.recording import Recording as GlassesRecording
from sqlalchemy import Column, String
from sqlalchemy.orm import Session, relationship

from src.config import RECORDINGS_PATH
from src.core.utils import download_file
from src.db.db import Base, engine



# SimRoom entity (defines a simulation room)
class SimRoom(Base):
    __tablename__ = "sim_rooms"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)

    # Add cascade="all, delete-orphan" to automatically delete related calibration recordings
    calibration_recordings = relationship("CalibrationRecording", back_populates="sim_room", cascade="all, delete-orphan")
    sim_room_classes = relationship("SimRoomClass", back_populates="sim_room", cascade="all, delete-orphan")

    @staticmethod
    def get_all() -> List["SimRoom"]:
        with Session(engine) as session:
            return session.query(SimRoom).all()
        
    @staticmethod
    def get(id: str) -> Union["SimRoom", None]:
        with Session(engine) as session:
            return session.query(SimRoom).filter(SimRoom.id == id).first()


# SimRoomClass entity (defines labeling classes for a SimRoom)
class SimRoomClass(Base):
    __tablename__ = "sim_room_classes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sim_room_id = Column(Integer, ForeignKey("sim_rooms.id"))
    class_name = Column(String)

    sim_room = relationship("SimRoom", back_populates="sim_room_classes")
    # Each class can be used in many annotations
    annotations = relationship("Annotation", back_populates="sim_room_class")

    @staticmethod
    def get(class_id: int) -> Union["SimRoomClass", None]:
        with Session(engine) as session:
            return session.query(SimRoomClass).filter(SimRoomClass.id == class_id).first()


# CalibrationRecording entity (links a Recording with a SimRoom for calibration/grounding)
class CalibrationRecording(Base):
    __tablename__ = "calibration_recordings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sim_room_id = Column(Integer, ForeignKey("sim_rooms.id"))
    recording_uuid = Column(String, ForeignKey("recordings.uuid"))

    sim_room = relationship("SimRoom", back_populates="calibration_recordings")
    recording = relationship("Recording", back_populates="calibration_recordings")
    # A CalibrationRecording can contain multiple annotations
    annotations = relationship("Annotation", back_populates="calibration_recording")

    def get_formatted(self) -> dict:
        return {
            "id": self.id,
            "participant": self.recording.participant,
            "created": self.recording.get_formatted()["created"]
        }

    @staticmethod
    def get_all() -> List["CalibrationRecording"]:
        with Session(engine) as session:
            return session.query(CalibrationRecording)\
                .options(joinedload(CalibrationRecording.recording))\
                .all()
        
    @staticmethod
    def get(id: str) -> Union["CalibrationRecording", None]:
        with Session(engine) as session:
            return session.query(CalibrationRecording)\
                .options(joinedload(CalibrationRecording.recording))\
                .filter(CalibrationRecording.id == id)\
                .first()

# Annotation entity (links a calibration recording with a specific SimRoomClass)
class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    calibration_recording_id = Column(Integer, ForeignKey("calibration_recordings.id"))
    sim_room_class_id = Column(Integer, ForeignKey("sim_room_classes.id"))

    calibration_recording = relationship("CalibrationRecording", back_populates="annotations")
    sim_room_class = relationship("SimRoomClass", back_populates="annotations")
    # Each annotation can have multiple point labels
    point_labels = relationship("PointLabel", back_populates="annotation")


# PointLabel entity (stores (x, y) coordinates and a binary label)
class PointLabel(Base):
    __tablename__ = "point_labels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    annotation_id = Column(Integer, ForeignKey("annotations.id"))
    x = Column(Float)
    y = Column(Float)
    label = Column(Boolean)  # true for positive segmentation label, false for negative

    annotation = relationship("Annotation", back_populates="point_labels")
