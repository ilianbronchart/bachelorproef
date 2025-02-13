from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from src.config import RECORDINGS_PATH

from .context import LabelingContext


class App(FastAPI):
    labeling_context: LabelingContext | None = None

    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)

        self.mount("/static", StaticFiles(directory="src/static", html=True), name="static")
        self.mount(str("/" / RECORDINGS_PATH), StaticFiles(directory=RECORDINGS_PATH), name="recordings")
