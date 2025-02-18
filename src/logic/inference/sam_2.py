from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from src.config import SAM_2_MODEL_CONFIGS
from torchvision.ops import masks_to_boxes


def load_sam2_predictor(checkpoint_path: Path) -> SAM2ImagePredictor:
    model_cfg = SAM_2_MODEL_CONFIGS[checkpoint_path]
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, str(checkpoint_path)))
    return predictor


def predict_sam2(
    predictor: SAM2ImagePredictor,
    points: list[tuple[int, int]],
    points_labels: list[int],
) -> tuple[npt.NDArray[np.uint8] | None, tuple[int, int, int, int] | None]:
    masks, _, _ = predictor.predict(
        point_coords=points,
        point_labels=points_labels,
        multimask_output=False,
    )

    if len(masks) == 0:
        return (None, None)

    mask = torch.from_numpy(masks[0]).unsqueeze(0)
    x1, y1, x2, y2 = masks_to_boxes(mask)[0].cpu().numpy().astype(np.int32)
    mask = mask.cpu().numpy().astype(np.uint8)

    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    final_mask = mask[:, y1:y2, x1:x2]

    return final_mask, (int(x1), int(y1), int(x2), int(y2))
