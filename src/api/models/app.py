from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from src.config import RECORDINGS_PATH

from .labeler import Labeler


class App(FastAPI):
    labeler: Labeler | None = None

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)

        self.mount(
            "/static", StaticFiles(directory="src/static", html=True), name="static"
        )
        self.mount(
            str("/" / RECORDINGS_PATH),
            StaticFiles(directory=RECORDINGS_PATH),
            name="recordings",
        )
