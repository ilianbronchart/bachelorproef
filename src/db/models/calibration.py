from typing import Union

from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Session, joinedload, relationship

from src.db.db import Base, engine


# SimRoom entity (defines a simulation room)
class SimRoom(Base):
    __tablename__ = "sim_rooms"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)

    # Add cascade="all, delete-orphan" to automatically delete related calibration recordings
    calibration_recordings = relationship(
        "CalibrationRecording", back_populates="sim_room", cascade="all, delete-orphan"
    )
    classes = relationship("SimRoomClass", back_populates="sim_room", cascade="all, delete-orphan")


# SimRoomClass entity (defines labeling classes for a SimRoom)
class SimRoomClass(Base):
    __tablename__ = "classes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sim_room_id = Column(Integer, ForeignKey("sim_rooms.id"))
    class_name = Column(String)
    color = Column(String, default="#198754")  # Default color is green

    sim_room = relationship("SimRoom", back_populates="classes")
    # Each class can be used in many annotations
    annotations = relationship("Annotation", back_populates="sim_room_class")

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.class_name, "color": self.color}


# CalibrationRecording entity (links a Recording with a SimRoom for calibration/grounding)
class CalibrationRecording(Base):
    __tablename__ = "calibration_recordings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sim_room_id = Column(Integer, ForeignKey("sim_rooms.id"))
    recording_uuid = Column(String, ForeignKey("recordings.uuid"))

    sim_room = relationship("SimRoom", back_populates="calibration_recordings")
    recording = relationship("Recording", back_populates="calibration_recordings")
    annotations = relationship("Annotation", back_populates="calibration_recording")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "participant": self.recording.participant,
            "created": self.recording.to_dict()["created"],
        }

    def save_annotations(self, frame_id: int, annotations_data: list[dict]) -> None:
        """Save or update annotations for this calibration recording"""
        with Session(engine) as session:
            # Delete existing annotations
            existing = session.query(Annotation).filter(Annotation.calibration_recording_id == self.id).all()
            for annotation in existing:
                session.delete(annotation)

            # Create new annotations
            for annotation_data in annotations_data:
                # Get sim room class by name
                sim_class = (
                    session.query(SimRoomClass)
                    .filter(
                        SimRoomClass.sim_room_id == self.sim_room_id,
                        SimRoomClass.class_name == annotation_data["label"],
                    )
                    .first()
                )

                if not sim_class:
                    continue  # Skip if class no longer exists

                annotation = Annotation(calibration_recording_id=self.id, sim_room_class_id=sim_class.id)
                session.add(annotation)
                session.flush()

                # Create point labels
                for point, label in zip(annotation_data["points"], annotation_data["point_labels"], strict=False):
                    point_label = PointLabel(annotation_id=annotation.id, x=point[0], y=point[1], label=(label == 1))
                    session.add(point_label)

            session.commit()

    def get_labeling_data(self) -> dict:
        """Get all data needed for the labeling view"""
        classes = [cls_.to_dict() for cls_ in self.sim_room.classes]
        # Set first class as active
        if classes:
            classes[0]["active"] = True

        return {
            "classes": classes,
            "annotations": [annotation.to_dict() for annotation in self.annotations],
            "calibration_recording": self,
            "recording_uuid": self.recording_uuid
        }

    @staticmethod
    def get_all() -> list["CalibrationRecording"]:
        with Session(engine) as session:
            return session.query(CalibrationRecording).options(joinedload(CalibrationRecording.recording)).all()

    @staticmethod
    def get(id: str) -> Union["CalibrationRecording", None]:
        with Session(engine) as session:
            return (
                session.query(CalibrationRecording)
                .options(joinedload(CalibrationRecording.recording))
                .filter(CalibrationRecording.id == id)
                .first()
            )


# Annotation entity (links a calibration recording with a specific SimRoomClass)
class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    calibration_recording_id = Column(Integer, ForeignKey("calibration_recordings.id"))
    sim_room_class_id = Column(Integer, ForeignKey("classes.id"))

    calibration_recording = relationship("CalibrationRecording", back_populates="annotations")
    sim_room_class = relationship("SimRoomClass", back_populates="annotations")
    # Each annotation can have multiple point labels
    point_labels = relationship("PointLabel", back_populates="annotation")

    def to_dict(self) -> dict:
        """Convert annotation to dictionary format needed for labeling view"""
        points = [(label.x, label.y) for label in self.point_labels]
        point_labels = [1 if label.label else 0 for label in self.point_labels]
        return {"label": self.sim_room_class.class_name, "points": points, "point_labels": point_labels}

    @staticmethod
    def create_from_dict(data: dict, calibration_recording_id: int, sim_class_id: int) -> "Annotation":
        """Create a new annotation from dictionary data"""
        annotation = Annotation(calibration_recording_id=calibration_recording_id, sim_room_class_id=sim_class_id)
        return annotation

    @classmethod
    def create_segmentation_result(cls, label: str, mask: str, point_labels: list[int]) -> dict:
        """Format segmentation result in a consistent way"""
        return {"class": label, "mask": mask, "point_labels": point_labels}


# PointLabel entity (stores (x, y) coordinates and a binary label)
class PointLabel(Base):
    __tablename__ = "point_labels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    annotation_id = Column(Integer, ForeignKey("annotations.id"))
    x = Column(Float)
    y = Column(Float)
    label = Column(Boolean)  # true for positive segmentation label, false for negative

    annotation = relationship("Annotation", back_populates="point_labels")
