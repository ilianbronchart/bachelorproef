from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi import Request as FastAPIRequest
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

DATA_PATH = Path("data/")
RECORDINGS_PATH = DATA_PATH / "recordings"
CHECKPOINTS_PATH = Path("checkpoints")
DEFAULT_GLASSES_HOSTNAME = "192.168.75.51"
FAST_SAM_CHECKPOINT = CHECKPOINTS_PATH / "FastSAM-x.engine"

templates = Jinja2Templates(directory="src/templates")


class App(FastAPI):
    is_inference_running: bool = False
    inference_model: Any | None = None  # TODO avoid Any

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mount("/static", StaticFiles(directory="src/static", html=True), name="static")
        self.mount(str("/" / RECORDINGS_PATH), StaticFiles(directory=RECORDINGS_PATH), name="recordings")


class Request(FastAPIRequest):
    app: App


@dataclass
class BaseContext:
    request: Request

    def to_dict(self):
        dict_ = {k: v for k, v in self.__dict__.items() if k != "request"}
        dict_["request"] = self.request
        return dict_


@dataclass(frozen=True)
class Template:
    # Pages
    INDEX: str = "index.jinja"
    RECORDINGS: str = "pages/recordings.jinja"
    SIMROOMS: str = "pages/simrooms.jinja"
    LABELER: str = "pages/labeler.jinja"

    # Components
    CONNECTION_STATUS: str = "components/connection-status.jinja"
    LOCAL_RECORDINGS: str = "components/local-recordings.jinja"
    GLASSES_RECORDINGS: str = "components/glasses-recordings.jinja"
    FAILED_CONNECTION: str = "components/failed-connection.jinja"
