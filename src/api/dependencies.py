from fastapi.responses import RedirectResponse
from src.api.models import Labeler, Request


def get_labeler(request: Request) -> Labeler | RedirectResponse:
    """
    Dependency that checks if the labeler exists in the app state.
    Returns the labeler if it exists, otherwise redirects to /simrooms.

    Use this as a dependency in routes that require an active labeler.
    """
    if request.app.labeler is None:
        return RedirectResponse(url="/simrooms", status_code=307)
    return request.app.labeler
