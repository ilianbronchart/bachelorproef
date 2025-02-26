from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, no_type_check

from fastapi import Request as FastAPIRequest
from src.config import Template
from src.db import Recording
from src.db.models import CalibrationRecording, SimRoom, SimRoomClass
from src.db.models.calibration import Annotation

if TYPE_CHECKING:
    from .app import App


class Request(FastAPIRequest):
    _app: "App"

    @property
    def app(self) -> "App":
        return self._app


@dataclass
class BaseContext:
    request: Request

    def to_dict(self, ignore: tuple[str] = ("request",)) -> dict[str, Any]:
        dict_ = {k: v for k, v in self.__dict__.items() if k not in ignore}
        dict_["request"] = self.request
        return dict_


@dataclass
class GlassesConnectionContext(BaseContext):
    glasses_connected: bool
    battery_level: float


@dataclass
class RecordingsContext(BaseContext):
    recordings: list[Recording] = field(default_factory=list)
    glasses_connected: bool = False
    failed_connection: bool = False
    content: str = Template.RECORDINGS


@dataclass
class SimRoomsContext(BaseContext):
    recordings: list[Recording] = field(default_factory=list)
    sim_rooms: list[SimRoom] = field(default_factory=list)
    selected_sim_room: SimRoom | None = None
    calibration_recordings: list[CalibrationRecording] = field(default_factory=list)
    classes: list[SimRoomClass] = field(default_factory=list)
    content: str = Template.SIMROOMS

    @no_type_check
    def to_dict(self) -> dict[str, Any]:
        dict_ = super().to_dict(ignore=["classes"])
        dict_["classes"] = [cls_.to_dict() for cls_ in self.classes]
        return dict_


@dataclass
class ClassListContext(BaseContext):
    selected_sim_room: SimRoom
    classes: list[SimRoomClass]

    @no_type_check
    def to_dict(self) -> dict[str, Any]:
        dict_ = super().to_dict(ignore=["classes"])
        dict_["classes"] = [cls_.to_dict() for cls_ in self.classes]
        return dict_


@dataclass
class LabelingContext(BaseContext):
    sim_room_id: int
    frame_count: int
    content: str = Template.LABELER


@dataclass
class LabelingAnnotationsContext(BaseContext):
    annotations: list[Annotation] = field(default_factory=list)
