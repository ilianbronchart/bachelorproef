from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class GazePoint:
    x: int
    y: int
    depth: float | None
    timestamp: float

    @property
    def position(self) -> tuple[int, int]:
        return (self.x, self.y)


@dataclass
class EyeGazeData:
    origin: npt.NDArray[np.float64]
    direction: npt.NDArray[np.float64]
    pupil_diameter: float


@dataclass(frozen=True)
class GazeDataType:
    GAZE: str = "gaze"
    MISSING: str = "missing"


@dataclass
class GazeData:
    type: str
    timestamp: float
    gaze2d: tuple[float, float] | None
    gaze3d: tuple[float, float, float] | None
    eye_data_left: EyeGazeData | None
    eye_data_right: EyeGazeData | None

    @staticmethod
    def from_dict(data: dict) -> "GazeData":  # type: ignore[type-arg]
        gaze_data = data["data"]

        if len(gaze_data.keys()) == 0:
            # No gaze data available for this timestamp
            return GazeData(
                type=GazeDataType.MISSING,
                timestamp=data["timestamp"],
                gaze2d=None,
                gaze3d=None,
                eye_data_left=None,
                eye_data_right=None,
            )

        left_eye = gaze_data.get("eyeleft", {})
        right_eye = gaze_data.get("eyeright", {})

        return GazeData(
            type=GazeDataType.GAZE,
            timestamp=data["timestamp"],
            gaze2d=tuple(gaze_data["gaze2d"]),
            gaze3d=tuple(gaze_data["gaze3d"]),
            eye_data_left=EyeGazeData(
                origin=np.array(left_eye["gazeorigin"]),
                direction=np.array(left_eye["gazedirection"]),
                pupil_diameter=left_eye["pupildiameter"],
            )
            if left_eye
            else None,
            eye_data_right=EyeGazeData(
                origin=np.array(right_eye["gazeorigin"]),
                direction=np.array(right_eye["gazedirection"]),
                pupil_diameter=right_eye["pupildiameter"],
            )
            if right_eye
            else None,
        )
