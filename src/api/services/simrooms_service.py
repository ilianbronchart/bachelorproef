from sqlalchemy.orm import Session

from src.api.models.pydantic import SimRoomClassDTO
from src.api.repositories import simrooms_repo
from src.config import LABELING_RESULTS_PATH


def get_class_id_to_name_map(db: Session, sim_room_id: int) -> dict[int, str]:
    """
    Get a mapping of class IDs to class names.
    """
    classes = simrooms_repo.get_simroom_classes(db, sim_room_id)
    return {simroom_class.id: simroom_class.class_name for simroom_class in classes}


def get_tracked_classes(db: Session, cal_rec_id: int) -> list[SimRoomClassDTO]:
    """
    Get all classes that have annotations for a given calibration recording.
    """
    classes = simrooms_repo.get_tracked_classes(db, cal_rec_id)
    return [SimRoomClassDTO.from_orm(simroom_class) for simroom_class in classes]


def get_annotation_paths(class_id: int):
    annotation_paths = []
    for labeling_results in LABELING_RESULTS_PATH.iterdir():
        for class_results in labeling_results.iterdir():
            if class_results.stem == str(class_id):
                for annotation in class_results.iterdir():
                    annotation_paths.append(annotation)
    return annotation_paths


def get_simroom_class(db: Session, class_id: int) -> SimRoomClassDTO:
    """
    Get a sim room class by its ID.
    """
    simroom_class = simrooms_repo.get_simroom_class(db, class_id)
    return SimRoomClassDTO.from_orm(simroom_class)
