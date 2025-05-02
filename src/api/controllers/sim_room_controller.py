from dataclasses import dataclass

from sqlalchemy.orm import Session

from src.api.db import engine
from src.api.models.db import SimRoomClass


@dataclass
class SimRoomClassResponse:
    id: int
    sim_room_id: int
    class_name: str
    color: str


def get_sim_room_classes(sim_room_id: int) -> list[SimRoomClassResponse]:
    """Get all classes for a given simulation room."""
    with Session(engine) as session:
        # Get the ids of all classes that have labels
        classes = (
            session.query(SimRoomClass)
            .filter(SimRoomClass.sim_room_id == sim_room_id)
            .all()
        )

        return [
            SimRoomClassResponse(
                id=sim_room_class.id,
                sim_room_id=sim_room_class.sim_room_id,
                class_name=sim_room_class.class_name,
                color=sim_room_class.color,
            )
            for sim_room_class in classes
        ]


def get_all_sim_room_classes() -> list[SimRoomClassResponse]:
    """Get all classes for all simulation rooms."""
    with Session(engine) as session:
        # Get the ids of all classes that have labels
        classes = session.query(SimRoomClass).all()

        return [
            SimRoomClassResponse(
                id=sim_room_class.id,
                sim_room_id=sim_room_class.sim_room_id,
                class_name=sim_room_class.class_name,
                color=sim_room_class.color,
            )
            for sim_room_class in classes
        ]
