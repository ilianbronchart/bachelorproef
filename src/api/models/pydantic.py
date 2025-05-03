import base64
import json
from datetime import datetime
from pathlib import Path

from g3pylib.recordings.recording import Recording as GlassesRecording
from pydantic import BaseModel, computed_field

from src.api.models.db import (
    Annotation as DBAnnotation,
)
from src.api.models.db import (
    CalibrationRecording as DBCalibrationRecording,
)
from src.api.models.db import (
    PointLabel as DBPointLabel,
)
from src.api.models.db import (
    Recording as DBRecording,
)
from src.api.models.db import (
    SimRoom as DBSimRoom,
)
from src.api.models.db import (
    SimRoomClass as DBSimRoomClass,
)


class BaseDTO(BaseModel):
    class Config:
        from_attributes = True


class RecordingDTO(BaseDTO):
    id: str
    visible_name: str
    participant: str
    created: datetime
    duration: str
    video_path: Path | None = None
    gaze_data_path: Path | None = None

    @computed_field(return_type=str)
    def formatted_duration(self) -> str:
        parts = self.duration.split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(float(parts[2]))
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @computed_field(return_type=str)
    def formatted_created(self) -> str:
        return self.created.strftime("%d/%m/%y at %I:%M %p")

    @classmethod
    async def parse_participant(recording: GlassesRecording) -> str:
        participant_bytes = base64.b64decode(await recording.meta_lookup("participant"))
        participant_json = participant_bytes.decode("utf-8")
        parsed_data = json.loads(participant_json)
        return parsed_data["name"] or "N/A"

    @classmethod
    def from_orm(cls, rec: DBRecording) -> "RecordingDTO":
        return cls(
            id=rec.id,
            visible_name=rec.visible_name,
            participant=rec.participant,
            created=datetime.fromisoformat(rec.created),
            duration=rec.duration,
            video_path=rec.video_path,
            gaze_data_path=rec.gaze_data_path,
        )

    @classmethod
    async def from_glasses_recording(
        cls, glasses_recording: GlassesRecording
    ) -> "RecordingDTO":
        return cls(
            id=glasses_recording.uuid,
            visible_name=await glasses_recording.get_visible_name(),
            participant=await cls.parse_participant(glasses_recording),
            created=await glasses_recording.get_created(),
            duration=str(await glasses_recording.get_duration()),
            video_path=None,
            gaze_data_path=None,
        )


class PointLabelDTO(BaseDTO):
    id: int
    annotation_id: int
    x: int
    y: int
    label: int

    @classmethod
    def from_orm(cls, point_label: DBPointLabel) -> "PointLabelDTO":
        return cls(
            id=point_label.id,
            annotation_id=point_label.annotation_id,
            x=point_label.x,
            y=point_label.y,
            label=point_label.label,
        )


class AnnotationDTO(BaseDTO):
    id: int
    calibration_recording_id: int
    sim_room_class_id: int
    frame_idx: int
    point_labels: list[PointLabelDTO]

    @classmethod
    def from_orm(cls, annotation: DBAnnotation) -> "AnnotationDTO":
        return cls(
            id=annotation.id,
            calibration_recording_id=annotation.calibration_recording_id,
            sim_room_class_id=annotation.sim_room_class_id,
            frame_idx=annotation.frame_idx,
            point_labels=[
                PointLabelDTO.from_orm(point_label)
                for point_label in annotation.point_labels
            ],
        )


class CalibrationRecordingDTO(BaseDTO):
    id: int
    sim_room_id: int
    recording_id: str
    recording: RecordingDTO
    annotations: list[AnnotationDTO]

    @classmethod
    def from_orm(
        cls, calibration_recording: DBCalibrationRecording
    ) -> "CalibrationRecordingDTO":
        return cls(
            id=calibration_recording.id,
            sim_room_id=calibration_recording.sim_room_id,
            recording_id=calibration_recording.recording_id,
            recording=RecordingDTO.from_orm(calibration_recording.recording),
            annotations=[
                AnnotationDTO.from_orm(annotation)
                for annotation in calibration_recording.annotations
            ],
            labeling_results_path=calibration_recording.labeling_results_path,
            annotations_path=calibration_recording.annotations_path,
            tracking_result_paths=calibration_recording.tracking_result_paths,
        )


class SimRoomClassDTO(BaseDTO):
    id: int
    sim_room_id: int
    class_name: str
    color: str

    @classmethod
    def from_orm(cls, sim_room_class: DBSimRoomClass) -> "SimRoomClassDTO":
        return cls(
            id=sim_room_class.id,
            sim_room_id=sim_room_class.sim_room_id,
            class_name=sim_room_class.class_name,
            color=sim_room_class.color,
            annotation_paths=[],
        )


class SimRoomDTO(BaseDTO):
    id: int
    name: str
    calibration_recordings: list[CalibrationRecordingDTO]
    classes: list[SimRoomClassDTO]

    @classmethod
    def from_orm(cls, sim_room: DBSimRoom) -> "SimRoomDTO":
        return cls(
            id=sim_room.id,
            name=sim_room.name,
            calibration_recordings=[
                CalibrationRecordingDTO.from_orm(cr)
                for cr in sim_room.calibration_recordings
            ],
            classes=[SimRoomClassDTO.from_orm(sc) for sc in sim_room.classes],
        )
