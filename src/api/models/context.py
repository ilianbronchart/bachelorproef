from typing import TYPE_CHECKING

from fastapi import Request as FastAPIRequest
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from src.api.models.pydantic import (
    AnnotationDTO,
    RecordingDTO,
    SimRoomClassDTO,
    SimRoomDTO,
)
from src.config import Template

if TYPE_CHECKING:
    from .app import App


class Request(FastAPIRequest):
    _app: "App"

    @property
    def app(self) -> "App":
        return self._app


class BaseContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Make request a private attribute (won't be serialized in `.dict()`)
    _request: Request = PrivateAttr()

    def __init__(self, **data):
        request = data.pop("_request", None)
        super().__init__(**data)
        self._request = request

    def model_dump(self, **kwargs):
        base = super().model_dump(**kwargs)
        base["request"] = self._request
        return base


class GlassesConnectionContext(BaseContext):
    glasses_connected: bool
    battery_level: float


class RecordingsContext(BaseContext):
    recordings: list[RecordingDTO] = Field(default_factory=list)
    glasses_connected: bool = False
    failed_connection: bool = False
    content: str = Template.RECORDINGS


class SimRoomsContext(BaseContext):
    recordings: list[RecordingDTO] = Field(default_factory=list)
    sim_rooms: list[SimRoomDTO] = Field(default_factory=list)
    selected_sim_room: SimRoomDTO | None = None
    content: str = Template.SIMROOMS


class ClassListContext(BaseContext):
    selected_sim_room: SimRoomDTO


class LabelingContext(BaseContext):
    simroom_id: int
    recording_id: str
    show_inactive_classes: bool
    content: str = Template.LABELER


class LabelingAnnotationsContext(BaseContext):
    annotations: list[AnnotationDTO]


class LabelingTimelineContext(BaseContext):
    frame_count: int
    current_frame_idx: int
    selected_class_id: int
    selected_class_color: str = "#000000"
    tracks: list[tuple[int, int]] = Field(default_factory=list)
    tracking_progress: float = 0.0
    is_tracking: bool = False
    update_canvas: bool = False


class LabelingClassesContext(BaseContext):
    selected_class_id: int
    simroom_id: int
    classes: list[SimRoomClassDTO]


class LabelingSettingsContext(BaseContext):
    show_inactive_classes: bool
