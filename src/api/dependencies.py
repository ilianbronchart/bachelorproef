from sqlalchemy.orm import Session

from src.api.db import engine


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
