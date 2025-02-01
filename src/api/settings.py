from dataclasses import asdict, dataclass
from typing import Any

import src.logic.glasses as glasses
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.core.config import RECORDINGS_PATH


@dataclass
class BaseContext:
    request: Request

    def to_dict(self):
        dict_ = {k: v for k, v in self.__dict__.items() if k != 'request'}
        dict_['request'] = self.request
        return dict_


class App(FastAPI):
    is_inference_running: bool = False
    inference_model: Any | None = None  # TODO avoid Any

    def __init__(self):
        super().__init__()

        self.mount("/static", StaticFiles(directory="src/static", html=True), name="static")
        self.mount(str("/" / RECORDINGS_PATH), StaticFiles(directory=RECORDINGS_PATH), name="recordings")

    def prepare(self):
        glasses.clean_local_recordings()


@dataclass(frozen=True)
class Template:
    # Pages
    INDEX: str = "index.jinja"
    RECORDINGS: str = "pages/recordings.jinja"
    LABELING: str = "pages/labeling.jinja"

    # Components
    CONNECTION_STATUS: str = "components/connection-status.jinja"
    LOCAL_RECORDINGS: str = "components/local-recordings.jinja"
    GLASSES_RECORDINGS: str = "components/glasses-recordings.jinja"
    LABELER: str = "components/labeler.jinja"
    FAILED_CONNECTION: str = "components/failed-connection.jinja"


app = App()
templates = Jinja2Templates(directory="src/templates")
