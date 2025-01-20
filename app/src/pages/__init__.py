from ..utils import Page
from .view_recordings import ViewRecordings

PAGE_MAP: dict[str, type[Page]] = {
    "View Recordings": ViewRecordings,
}

__all__ = ["PAGE_MAP"]
