from dataclasses import dataclass
from pathlib import Path

from fastapi.templating import Jinja2Templates

DATA_PATH = Path("data/")
RECORDINGS_PATH = DATA_PATH / "recordings"
FRAMES_PATH = DATA_PATH / "frames"
CHECKPOINTS_PATH = Path("checkpoints")
DEFAULT_GLASSES_HOSTNAME = "192.168.75.51"
FAST_SAM_CHECKPOINT = CHECKPOINTS_PATH / "FastSAM-x.engine"

templates = Jinja2Templates(directory="src/templates")


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
    CLASS_LIST: str = "components/class-list.jinja"
