from pathlib import Path

import numpy as np
import torch
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from torchvision.ops import masks_to_boxes

from src.aliases import Int32Array, UInt8Array
from src.api.exceptions import PredictionFailedError
from src.config import MAX_INFERENCE_STATE_FRAMES, SAM_2_MODEL_CONFIGS


def load_predictor(checkpoint_path: Path) -> SAM2ImagePredictor:
    model_cfg = SAM_2_MODEL_CONFIGS[checkpoint_path]
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, str(checkpoint_path)))
    return predictor


def load_video_predictor(
    checkpoint_path: Path, max_inference_state_frames: int = MAX_INFERENCE_STATE_FRAMES
) -> SAM2VideoPredictor:
    model_cfg = SAM_2_MODEL_CONFIGS[checkpoint_path]
    print(checkpoint_path, model_cfg)
    predictor = build_sam2_video_predictor(
        model_cfg,
        str(checkpoint_path),
        device="cuda",
        max_cond_frames_in_attn=max_inference_state_frames,
        clear_non_cond_mem_around_input=True,
    )
    return predictor


def predict(
    predictor: SAM2ImagePredictor,
    points: list[tuple[int, int]],
    points_labels: list[int],
) -> tuple[UInt8Array, Int32Array]:
    masks, _, _ = predictor.predict(
        point_coords=np.array(points),
        point_labels=np.array(points_labels),
        multimask_output=False,
    )

    if len(masks) == 0:
        raise PredictionFailedError("No masks found for the given points")

    mask_torch = torch.from_numpy(masks[0]).unsqueeze(0)
    x1, y1, x2, y2 = masks_to_boxes(mask_torch)[0].cpu().numpy().astype(np.int32)
    mask = mask_torch.cpu().numpy().astype(np.uint8)

    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    final_mask = mask[:, y1:y2, x1:x2]

    return final_mask, np.array([x1, y1, x2, y2]).astype(np.int32)
