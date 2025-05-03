from typing import TYPE_CHECKING

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.api.db import Base

# Workaround for circular imports due to type references
if TYPE_CHECKING:
    from .calibration import CalibrationRecording


class Recording(Base):
    __tablename__ = "recordings"

    uuid: Mapped[str] = mapped_column(String, primary_key=True)
    visible_name: Mapped[str] = mapped_column(String)
    participant: Mapped[str] = mapped_column(String)
    created: Mapped[str] = mapped_column(String)
    duration: Mapped[str] = mapped_column(String)
    folder_name: Mapped[str] = mapped_column(String)

    calibration_recordings: Mapped[list["CalibrationRecording"]] = relationship(
        "CalibrationRecording", back_populates="recording"
    )
