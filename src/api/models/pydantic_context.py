from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastapi import Request as FastAPIRequest
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr

from src.api.models.pydantic import RecordingDTO
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
        request = data.pop('_request', None)
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
