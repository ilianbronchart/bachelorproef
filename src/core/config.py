from pathlib import Path

DATA_PATH = "data/"
RECORDINGS_PATH = Path(DATA_PATH) / "recordings"

DEFAULT_GLASSES_HOSTNAME = "192.168.75.51"

Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
Path(RECORDINGS_PATH).mkdir(parents=True, exist_ok=True)
