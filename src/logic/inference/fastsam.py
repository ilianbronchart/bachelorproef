from pathlib import Path
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt
import torch
from src.logic.glasses.domain import GazeData, GazePoint
from src.logic.glasses.gaze import get_gaze_points, match_frames_to_gaze
from src.utils import cv2_loadvideo
from torchvision.ops import masks_to_boxes
from ultralytics import FastSAM
from ultralytics.engine.results import Boxes, Masks, Results


def overlay_gaze_points(frame, gaze_points: list[GazePoint], radius: int):
    """
    Overlay gaze points on the original frame

    Args:
        frame: Original frame (ndarray)
        gaze_points: List of gaze points (x, y) in pixel coordinates

    Returns:
        frame_with_gazepoints: Frame with overlaid gaze points
    """
    for gaze_point in gaze_points:
        cv2.circle(frame, gaze_point.position, radius, (255, 0, 0), 2)


def recalculate_boxes(results: Results):
    """
    Recalculate boxes from masks

    Args:
        results: Results object from FastSAM
    """
    if results.masks is None:
        return results

    boxes = torch.stack([
        torch.cat([masks_to_boxes(mask.data)[0], results.boxes[i].conf, results.boxes[i].cls])
        for i, mask in enumerate(results.masks)
    ])

    results.update(boxes=boxes)


def filter_result(results: Results, filtered_idxs: list[int]) -> None:
    """
    Filter out boxes and masks from the results

    Args:
        results: Results object from FastSAM
        filtered_idxs: List of indices to filter out
    """
    if len(filtered_idxs) == 0:
        results.boxes = None
        results.masks = None
        return results

    boxes = torch.stack([
        torch.cat([results.boxes[i].xyxy[0], results.boxes[i].conf, results.boxes[i].cls]) for i in filtered_idxs
    ])

    masks = torch.stack([results.masks[i].data[0] for i in filtered_idxs])

    results.update(boxes=boxes, masks=masks)


def filter_large_masks(results: Results) -> None:
    """
    Filter out masks with area greater than 60% of the frame area

    Args:
        results: Results object from FastSAM
    """
    if results.masks is None:
        return results

    # Assuming all masks have the same shape
    sample_mask = results.masks[0].data
    _, height, width = sample_mask.shape  # Assuming mask shape is (1, H, W)

    frame_area = height * width
    max_mask_area = 0.3 * frame_area

    filtered_masks = []
    for i, mask in enumerate(results.masks):
        mask_area = mask.data.sum()
        if mask_area <= max_mask_area:
            filtered_masks.append(i)

    filter_result(results, filtered_masks)


def filter_viewed_masks(results: Results, gaze_point: tuple[int, int], viewed_radius: int) -> None:
    """
    Filter out masks that are not within the viewed radius of the gaze point

    Args:
        results: Results object from FastSAM
        gaze_point: Gaze point (x, y) in pixel coordinates
        viewed_radius: Radius in pixels

    Returns:
        results: Results object with filtered masks and boxes
    """
    if results.masks is None:
        return results

    sample_mask = results.masks[0].data
    _, height, width = sample_mask.shape
    device = sample_mask.device

    # Create a circular mask centered at the gaze point
    y = torch.arange(0, height, device=device).view(-1, 1).repeat(1, width)
    x = torch.arange(0, width, device=device).view(1, -1).repeat(height, 1)
    dist_sq = (x - gaze_point[0]) ** 2 + (y - gaze_point[1]) ** 2
    circular_mask = (dist_sq <= viewed_radius**2).float().unsqueeze(0)  # Shape: (1, H, W)

    viewed_masks = []
    for i, mask in enumerate(results.masks):
        overlap = (mask.data * circular_mask).sum()
        if overlap > 0:
            viewed_masks.append(i)

    filter_result(results, viewed_masks)


def get_viewed_masks(
    video_path: Path,
    model: FastSAM,
    crop_size: int,
    resolution: tuple[int, int],
    frame_count: int,
    fps: float,
    gaze_data: list[GazeData],
    conf: float,
    iou: float,
    viewed_radius: int,
    save_video=False,
) -> list[Results]:
    half_crop = crop_size // 2
    gaze_points = get_gaze_points(gaze_data, resolution)
    frame_gaze_mapping = match_frames_to_gaze(frame_count, gaze_points, fps)

    if save_video:
        video_result = cv2.VideoWriter(
            # get file name from video path (only the filename not the path)
            video_path.name,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (resolution[1], resolution[0]),
        )

    results_per_frame = []
    for frame_idx, frame in cv2_loadvideo(video_path):
        if save_video:
            original_frame = frame.copy()

        frame_gaze_points = frame_gaze_mapping[frame_idx]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if len(frame_gaze_points) == 0:
            # No gaze points for this frame
            results = Results(orig_img=torch.tensor([]), path="", names="")
            results_per_frame.append(results)
            if save_video:
                video_result.write(original_frame)
            continue

        # Crop the frame around the gaze point
        gaze_point = frame_gaze_points[0]
        cx, cy = gaze_point.position

        left = max(cx - half_crop, 0)
        right = min(cx + half_crop, resolution[1])
        top = max(cy - half_crop, 0)
        bottom = min(cy + half_crop, resolution[0])

        crop_x = min(cx - left, half_crop - 1)
        crop_y = min(cy - top, half_crop - 1)
        cropped_frame = frame[top:bottom, left:right]

        # Pad the bottom right corner to make the frame square
        pad_x = crop_size - cropped_frame.shape[1]
        pad_y = crop_size - cropped_frame.shape[0]
        padded_frame = cv2.copyMakeBorder(cropped_frame, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        results: Results = model(
            source=padded_frame, retina_masks=True, device="cuda", verbose=False, imgsz=crop_size, conf=conf, iou=iou
        )[0]

        filter_viewed_masks(results, (crop_x, crop_y), viewed_radius)
        filter_large_masks(results)
        recalculate_boxes(results)  # Seems like some boxes are wrong so we need to recalculate them
        results_per_frame.append(results)

        if save_video:
            result_frame = results.plot(conf=True)
            result_unpad = result_frame[0 : padded_frame.shape[0] - pad_y, 0 : padded_frame.shape[1] - pad_x]
            original_frame[top:bottom, left:right] = cv2.cvtColor(result_unpad, cv2.COLOR_BGR2RGB)
            overlay_gaze_points(original_frame, frame_gaze_points, viewed_radius)
            video_result.write(original_frame)

    if save_video:
        video_result.release()

    return results_per_frame


def crop_box(orig_img: np.ndarray, box: Boxes, mask: Masks | None = None) -> np.ndarray:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

    if mask is not None:
        mask_np = mask.data.permute(1, 2, 0).cpu().numpy()
        mask_np = np.repeat(mask_np, 3, axis=2)

        if orig_img.dtype != mask_np.dtype:
            mask_np = mask_np.astype(orig_img.dtype)

        masked_img = orig_img * mask_np
        return masked_img[y1:y2, x1:x2]

    return orig_img[y1:y2, x1:x2]


def segment(
    image: npt.NDArray[np.uint8],
    model: FastSAM,
    points: list[tuple[int, int]],
    point_labels: list[int],
) -> tuple[npt.NDArray[np.uint8] | None, tuple[int, int, int, int] | None]:
    """
    Perform inference on image using points

    Args:
        image: Image to perform inference on. (BGR)
        model: FastSAM model
        points: List of points to use for inference
        point_labels: List of labels for each point (1 for positive 0 for negative)
        conf: Confidence threshold
        iou: IoU threshold

    Returns:
        torch.Tensor: Merged mask
    """
    cx, cy = np.mean(points, axis=0)
    crop_size = 512
    half_crop = crop_size // 2
    resolution = image.shape[:2]

    left = int(max(cx - half_crop, 0))
    right = int(min(cx + half_crop, resolution[1]))
    top = int(max(cy - half_crop, 0))
    bottom = int(min(cy + half_crop, resolution[0]))

    cropped_frame = image[top:bottom, left:right]
    points = [(x - left, y - top) for x, y in points]

    # Pad the bottom right corner to make the frame square
    pad_x = crop_size - cropped_frame.shape[1]
    pad_y = crop_size - cropped_frame.shape[0]
    padded_frame = cv2.copyMakeBorder(cropped_frame, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    padded_frame = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB)

    results: Results = cast(
        Results,
        model(
            source=padded_frame,
            points=points,
            labels=point_labels,
            device="cuda",
            verbose=False,
            imgsz=crop_size,
        )[0],
    )

    if results.masks is None or len(results.masks) == 0:
        return (None, None)

    merged_mask = torch.zeros_like(cast(torch.Tensor, results.masks[0].data))
    for mask in results.masks:
        merged_mask = torch.logical_or(merged_mask, cast(torch.Tensor, mask.data))

    x1, y1, x2, y2 = masks_to_boxes(merged_mask)[0].cpu().numpy().astype(np.int32)
    merged_mask = merged_mask.cpu().numpy().astype(np.uint8)

    # If the coordinates are swapped, correct them:
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    final_mask = merged_mask[:, y1:y2, x1:x2]

    # Adjust the bounding box to the original image coordinates:
    x1 += left
    y1 += top
    x2 += left
    y2 += top

    return final_mask, (int(x1), int(y1), int(x2), int(y2))
