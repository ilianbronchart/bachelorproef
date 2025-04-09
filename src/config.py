import logging
import os
from dataclasses import dataclass
from pathlib import Path

import dotenv
from fastapi.templating import Jinja2Templates

# Load environment variables from .env file
dotenv.load_dotenv(".env")

SRC_PATH = Path(os.environ.get("SRC_PATH", "src"))
DATA_PATH = Path("data/")
RECORDINGS_PATH = DATA_PATH / "recordings"
RECORDINGS_PATH.mkdir(exist_ok=True)
CHECKPOINTS_PATH = Path(os.environ.get("CHECKPOINTS_PATH", "checkpoints"))
CHECKPOINTS_PATH.mkdir(exist_ok=True)
LABELING_RESULTS_PATH = DATA_PATH / "labeling_results"
LABELING_RESULTS_PATH.mkdir(exist_ok=True)
LABELING_ANNOTATIONS_DIR = "annotations"
STATIC_FILES_PATH = SRC_PATH / "static"
TEMPLATES_PATH = SRC_PATH / "templates"

DEFAULT_GLASSES_HOSTNAME = "192.168.75.51"
FAST_SAM_CHECKPOINT = CHECKPOINTS_PATH / "FastSAM-x.pt"

# The amount of frames kept in memory for SAM2 video inference
MAX_INFERENCE_STATE_FRAMES = 100

# Gaze Segmentation parameters:
TOBII_FOV_X = 95
GAZE_FOVEA_FOV = 2
TOBII_GLASSES_FPS = 24.95
TOBII_GLASSES_RESOLUTION = (1920, 1080)
VIEWED_RADIUS = int(GAZE_FOVEA_FOV / TOBII_FOV_X * TOBII_GLASSES_RESOLUTION[0] / 2)
UNKNOWN_CLASS_ID = -1


@dataclass(frozen=True)
class Sam2Checkpoints:
    BASE_PLUS: Path = CHECKPOINTS_PATH / "sam2.1_hiera_base_plus.pt"
    LARGE: Path = CHECKPOINTS_PATH / "sam2.1_hiera_large.pt"
    SMALL: Path = CHECKPOINTS_PATH / "sam2.1_hiera_small.pt"
    TINY: Path = CHECKPOINTS_PATH / "sam2.1_hiera_tiny.pt"


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
    LABELING_CONTROLS: str = "components/labeling/labeling-controls.jinja"
    LABELING_SETTINGS: str = "components/labeling/labeling-settings.jinja"


class EndpointFilter(logging.Filter):
    # List of endpoints to filter out from logs
    FILTERED_ENDPOINTS = ["/labeling/controls", "/glasses/connection"]

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(endpoint in message for endpoint in self.FILTERED_ENDPOINTS)


logging.getLogger("uvicorn.access").addFilter(EndpointFilter())
