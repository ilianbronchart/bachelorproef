from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.config import RECORDINGS_PATH, STATIC_FILES_PATH
from src.api.services.labeling_service import Labeler


class App(FastAPI):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.labeler: Labeler | None = None

        self.mount(
            "/static",
            StaticFiles(directory=str(STATIC_FILES_PATH), html=True),
            name="static",
        )
        self.mount(
            str("/" / RECORDINGS_PATH),
            StaticFiles(directory=RECORDINGS_PATH),
            name="recordings",
        )
