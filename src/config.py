import logging
import os
from dataclasses import dataclass
from pathlib import Path

from fastapi.templating import Jinja2Templates

SRC_PATH = Path("src")
DATA_PATH = Path("data/")
DATA_PATH.mkdir(exist_ok=True)
RECORDINGS_PATH = DATA_PATH / "recordings"
RECORDINGS_PATH.mkdir(exist_ok=True)
CHECKPOINTS_PATH = Path(os.environ.get("CHECKPOINTS_PATH", "checkpoints"))
CHECKPOINTS_PATH.mkdir(exist_ok=True)
TRACKING_RESULTS_PATH = Path(
    os.environ.get("TRACKING_RESULTS_PATH", DATA_PATH / "labeling_results")
)
TRACKING_RESULTS_PATH.mkdir(exist_ok=True)
STATIC_FILES_PATH = SRC_PATH / "static"
TEMPLATES_PATH = SRC_PATH / "templates"
DEFAULT_GLASSES_HOSTNAME = "192.168.75.51"
FAST_SAM_CHECKPOINT = CHECKPOINTS_PATH / "FastSAM-x.pt"
DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"

if not FAST_SAM_CHECKPOINT.exists():
    raise FileNotFoundError(f"FastSAM checkpoint not found at {FAST_SAM_CHECKPOINT}.")

# The amount of frames kept in memory for SAM2 video inference
MAX_INFERENCE_STATE_FRAMES = 100

# Gaze Segmentation parameters:
TOBII_FOV_X = 95
GAZE_FOV = 1 + 0.6  # 1 degree fovea + 0.6 degree eyetracker accuracy
TOBII_GLASSES_RESOLUTION = (1080, 1920)
VIEWED_RADIUS = int(GAZE_FOV / TOBII_FOV_X * TOBII_GLASSES_RESOLUTION[1] / 2)
TOBII_GLASSES_FPS = 24.95

@dataclass(frozen=True)
class Sam2Checkpoints:
    BASE_PLUS: Path = CHECKPOINTS_PATH / "sam2.1_hiera_base_plus.pt"
    LARGE: Path = CHECKPOINTS_PATH / "sam2.1_hiera_large.pt"
    SMALL: Path = CHECKPOINTS_PATH / "sam2.1_hiera_small.pt"
    TINY: Path = CHECKPOINTS_PATH / "sam2.1_hiera_tiny.pt"


for checkpoint in Sam2Checkpoints.__dict__.values():
    if isinstance(checkpoint, Path) and not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint}. Please download the model."
        )

SAM_2_MODEL_CONFIGS = {
    Sam2Checkpoints.BASE_PLUS: "sam2.1_hiera_b+.yaml",
    Sam2Checkpoints.LARGE: "sam2.1_hiera_l.yaml",
    Sam2Checkpoints.SMALL: "sam2.1_hiera_s.yaml",
    Sam2Checkpoints.TINY: "sam2.1_hiera_t.yaml",
}

templates = Jinja2Templates(directory=str(TEMPLATES_PATH))


@dataclass(frozen=True)
class Template:
    # Pages
    INDEX: str = "index.jinja"
    RECORDINGS: str = "pages/recordings.jinja"
    SIMROOMS: str = "pages/simrooms.jinja"
    LABELER: str = "pages/labeling.jinja"

    # Components
    CONNECTION_STATUS: str = "components/connection-status.jinja"
    LOCAL_RECORDINGS: str = "components/local-recordings.jinja"
    GLASSES_RECORDINGS: str = "components/glasses-recordings.jinja"
    CLASS_LIST: str = "components/class-list.jinja"

    # Labeling
    LABELING_CLASSES: str = "components/labeling/labeling-classes.jinja"
    LABELING_ANNOTATIONS: str = "components/labeling/labeling-annotations.jinja"
    LABELING_CANVAS: str = "components/labeling/labeling-canvas.jinja"
    LABELING_TIMELINE: str = "components/labeling/labeling-timeline.jinja"
    LABELING_SETTINGS: str = "components/labeling/labeling-settings.jinja"


class EndpointFilter(logging.Filter):
    # List of endpoints to filter out from logs
    FILTERED_ENDPOINTS = ["/labeling/timeline", "/glasses/connection"]

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(endpoint in message for endpoint in self.FILTERED_ENDPOINTS)


logging.getLogger("uvicorn.access").addFilter(EndpointFilter())
