from sqlalchemy.orm import Session

from src.api.db import engine
from src.api.exceptions import LabelingServiceNotAvailableError
from src.api.models import Request
from src.api.services.labeling_service import Labeler


def get_db():
    db = Session(engine)
    try:
        yield db
        db.commit()
    except:
        db.rollback()
        raise
    finally:
        db.close()


def require_labeler(request: Request) -> Labeler:
    if request.app.labeler is None:
        raise LabelingServiceNotAvailableError()
    return request.app.labeler
