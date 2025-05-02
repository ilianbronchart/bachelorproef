from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from src.api.models import Labeler, Request
from src.api.db import engine
from src.api.models.db.calibration import SimRoomClass


def get_labeler(request: Request) -> Labeler | RedirectResponse:
    """
    Dependency that checks if the labeler exists in the app state.
    Returns the labeler if it exists, otherwise redirects to /simrooms.

    Use this as a dependency in routes that require an active labeler.
    """
    if request.app.labeler is None:
        return RedirectResponse(url="/simrooms", status_code=307)
    return request.app.labeler


def get_selected_class_id(request: Request) -> int | RedirectResponse:
    """
    Dependency that gets the selected class id from the labeler.
    """
    if request.app.labeler is None:
        return RedirectResponse(url="/simrooms", status_code=307)

    labeler = request.app.labeler
    with Session(engine) as session:
        classes = (
            session.query(SimRoomClass)
            .filter(SimRoomClass.sim_room_id == labeler.calibration_recording.sim_room_id)
            .all()
        )

        ids = [cls_.id for cls_ in classes]
        if labeler.selected_class_id not in ids:
            return -1

        return labeler.selected_class_id
