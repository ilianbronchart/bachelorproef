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
    video_path: Path = Path("N/A")
    gaze_data_path: Path = Path("N/A")

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
    async def parse_participant(cls, recording: GlassesRecording) -> str:
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
    calibration_id: int
    simroom_class_id: int
    frame_idx: int
    simroom_class: "SimRoomClassDTO"
    point_labels: list[PointLabelDTO]
    mask_base64: str
    frame_crop_base64: str
    box: tuple[int, int, int, int]

    @classmethod
    def from_orm(cls, annotation: DBAnnotation) -> "AnnotationDTO":
        return cls(
            id=annotation.id,
            calibration_id=annotation.calibration_id,
            simroom_class_id=annotation.simroom_class_id,
            frame_idx=annotation.frame_idx,
            simroom_class=SimRoomClassDTO.from_orm(annotation.simroom_class),
            point_labels=[
                PointLabelDTO.from_orm(point_label)
                for point_label in annotation.point_labels
            ],
            mask_base64=annotation.mask_base64,
            frame_crop_base64=annotation.frame_crop_base64,
            box=tuple(json.loads(annotation.box_json)),
        )


class CalibrationRecordingDTO(BaseDTO):
    id: int
    simroom_id: int
    recording_id: str
    recording: RecordingDTO
    annotations: list[AnnotationDTO]
    video_path: Path
    tracking_results_path: Path
    tracking_result_paths: list[Path]

    @classmethod
    def from_orm(
        cls, calibration_recording: DBCalibrationRecording
    ) -> "CalibrationRecordingDTO":
        return cls(
            id=calibration_recording.id,
            simroom_id=calibration_recording.simroom_id,
            recording_id=calibration_recording.recording_id,
            recording=RecordingDTO.from_orm(calibration_recording.recording),
            annotations=[
                AnnotationDTO.from_orm(annotation)
                for annotation in calibration_recording.annotations
            ],
            video_path=calibration_recording.video_path,
            tracking_results_path=calibration_recording.tracking_results_path,
            tracking_result_paths=calibration_recording.tracking_result_paths,
        )


class SimRoomClassDTO(BaseDTO):
    id: int
    simroom_id: int
    class_name: str
    color: str

    @classmethod
    def from_orm(cls, simroom_class: DBSimRoomClass) -> "SimRoomClassDTO":
        return cls(
            id=simroom_class.id,
            simroom_id=simroom_class.simroom_id,
            class_name=simroom_class.class_name,
            color=simroom_class.color,
        )


class SimRoomDTO(BaseDTO):
    id: int
    name: str
    calibration_recordings: list[CalibrationRecordingDTO]
    classes: list[SimRoomClassDTO]

    @classmethod
    def from_orm(cls, simroom: DBSimRoom) -> "SimRoomDTO":
        return cls(
            id=simroom.id,
            name=simroom.name,
            calibration_recordings=[
                CalibrationRecordingDTO.from_orm(cr)
                for cr in simroom.calibration_recordings
            ],
            classes=[SimRoomClassDTO.from_orm(sc) for sc in simroom.classes],
        )
