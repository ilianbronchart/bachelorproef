from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from src.db import engine
from src.db.models import CalibrationRecording, SimRoomClass


@dataclass
class AnnotatedClassResponse:
    id: int
    sim_room_id: int
    class_name: str
    color: str
    annotation_paths: list[Path]


def get_annotated_classes(cal_rec_id: int) -> list[AnnotatedClassResponse]:
    """Get all classes that have annotations for a given calibration recording."""

    with Session(engine) as session:
        # Get the calibration recording
        cal_rec: CalibrationRecording | None = session.query(CalibrationRecording).get(
            cal_rec_id
        )
        if not cal_rec:
            raise ValueError(f"Calibration recording with id {cal_rec_id} not found")

        # Get the ids of all classes that have labels
        result_paths = cal_rec.labeling_result_paths
        result_paths = [path for path in result_paths if len(list(path.iterdir())) > 0]
        try:
            class_ids = [int(path.stem) for path in result_paths]
        except Exception as e:
            raise ValueError(f"Unexpected Error occured: {e}")

        classes = session.query(SimRoomClass).filter(SimRoomClass.id.in_(class_ids)).all()

        return [
            AnnotatedClassResponse(
                id=sim_room_class.id,
                sim_room_id=sim_room_class.sim_room_id,
                class_name=sim_room_class.class_name,
                color=sim_room_class.color,
                annotation_paths=list(
                    result_paths[class_ids.index(sim_room_class.id)].iterdir()
                ),
            )
            for sim_room_class in classes
        ]


def get_recording_path(cal_rec_id: int) -> Path:
    """Get the path of the recording for a given calibration recording."""
    with Session(engine) as session:
        cal_rec: CalibrationRecording | None = session.query(CalibrationRecording).get(
            cal_rec_id
        )
        if not cal_rec:
            raise ValueError(f"Calibration recording with id {cal_rec_id} not found")

        return cal_rec.recording.video_path


def get_gaze_data_path(cal_rec_id: int) -> Path:
    """Get the path of the gaze data for a given calibration recording."""
    with Session(engine) as session:
        cal_rec: CalibrationRecording | None = session.query(CalibrationRecording).get(
            cal_rec_id
        )
        if not cal_rec:
            raise ValueError(f"Calibration recording with id {cal_rec_id} not found")

        return cal_rec.recording.gaze_data_path


def recording_uuid_to_calibration_id(recording_uuid: str) -> int:
    """Get the calibration recording id from the recording uuid."""
    with Session(engine) as session:
        cal_rec: CalibrationRecording | None = (
            session.query(CalibrationRecording)
            .filter(CalibrationRecording.recording_uuid == recording_uuid)
            .first()
        )
        if not cal_rec:
            raise ValueError(
                f"Calibration recording with uuid {recording_uuid} not found"
            )

        return cal_rec.id


def get_calibration_recording_by_uuid(
    recording_uuid: str,
) -> CalibrationRecording | None:
    """Get the calibration recording by its uuid."""
    with Session(engine) as session:
        cal_rec: CalibrationRecording | None = (
            session.query(CalibrationRecording)
            .filter(CalibrationRecording.recording_uuid == recording_uuid)
            .first()
        )
        return cal_rec
