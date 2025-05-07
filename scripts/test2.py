import numpy as np
import cv2
from src.api.db import engine, Session
from src.api.repositories import annotations_repo
from src.api.utils import image_utils

with Session(engine) as session:
    calibration_id = 3

    annotations = annotations_repo.get_annotations_by_frame_idx(
        db=session,
        calibration_id=calibration_id,
        frame_idx=1818,
    )

    for i, anno in enumerate(annotations):
        mask = image_utils.decode_from_base64(anno.mask_base64) * 255

        path = f"scripts/test_{calibration_id}_{anno.frame_idx}_{anno.simroom_class_id}.png"
        # save result_img as png

        cv2.imwrite(path, mask)

        # save anno.frame_crop_base64 as png