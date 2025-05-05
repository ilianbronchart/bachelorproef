import base64
from typing import cast

import cv2
import numpy as np
from PIL import ImageColor
from src.aliases import UInt8Array
from src.api.exceptions import ImageEncodingError


def draw_mask(
    img: UInt8Array, mask: UInt8Array, box: tuple[int, int, int, int]
) -> UInt8Array:
    x1, y1, x2, y2 = box
    roi = img[y1:y2, x1:x2].copy()
    white_overlay = np.full(roi.shape, 255, dtype=np.uint8)
    mask = mask.astype(bool)

    # Apply alpha blending (0.2) on the white overlay at mask locations
    roi[mask] = cv2.addWeighted(roi[mask], 1, white_overlay[mask], 0.2, 0)
    img[y1:y2, x1:x2] = roi
    return img


def draw_labeled_box(
    img: UInt8Array, box: tuple[int, int, int, int], label: str, color: str
) -> UInt8Array:
    x1, y1, x2, y2 = box
    color_rgb = cast(tuple[int, int, int], ImageColor.getcolor(color, "RGB"))
    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)

    label_bg_y1 = max(y1 - text_h - 10, 0)
    label_bg_x2 = x1 + text_w + 10

    cv2.rectangle(
        img,
        (x1 - 1, label_bg_y1),  # Top-left corner
        (label_bg_x2, y1),  # Bottom-right corner
        color_bgr,
        -1,  # Filled rectangle
    )
    cv2.putText(
        img,
        label,
        (x1 + 5, y1 - 7),  # Original text position relative to y1
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),  # Black text color
        2,
    )

    return img


def encode_to_png_bytes(image: UInt8Array) -> bytes:
    """Encode an image to PNG format and return as bytes."""
    ret, encoded_img = cv2.imencode(".png", image)
    if not ret:
        raise ImageEncodingError("Failed to encode image")
    return encoded_img.tobytes()


def encode_to_png(image: UInt8Array) -> str:
    """Encode an image to PNG format and return as base64 string."""
    ret, encoded_img = cv2.imencode(".png", image)
    if not ret:
        raise ImageEncodingError("Failed to encode image")
    return base64.b64encode(encoded_img.tobytes()).decode("utf-8")


def decode_from_base64(base64_str: str) -> UInt8Array:
    """Decode a base64 string to an image."""
    try:
        decoded_data = base64.b64decode(base64_str)
        np_arr = np.frombuffer(decoded_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED).astype(np.uint8)
    except Exception as e:
        raise ImageEncodingError(f"Failed to decode image: {e}") from e

    return img
