import base64

import cv2
import numpy as np
from PIL import ImageColor
from src.aliases import UInt8Array
from src.api.exceptions import ImageEncodingError


def draw_mask(img: UInt8Array, mask: UInt8Array, box: tuple[int, int, int, int]) -> UInt8Array:
    x1, y1, x2, y2 = box

    roi = img[y1:y2, x1:x2].copy()
    white_overlay = np.full(roi.shape, 255, dtype=np.uint8)
    mask = mask.astype(bool)

    # Apply alpha blending (0.2) on the white overlay only at locations where the mask is present
    roi[mask] = cv2.addWeighted(
        roi[mask], 1, white_overlay[mask], 0.2, 0
    )
    img[y1:y2, x1:x2] = roi
    return img


def draw_box(img: UInt8Array, box: tuple[int, int, int, int], label: str, color: str) -> UInt8Array:
    x1, y1, x2, y2 = box
    color_rgb = ImageColor.getcolor(color, "RGB")
    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])  # type: ignore[index]

    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
    )
    cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)
    cv2.rectangle(
        img,
        (x1, y1 - text_height - 10),
        (x1 + text_width + 10, y1),
        color_bgr,
        -1,
    )
    cv2.putText(
        img,
        label,
        (x1 + 5, y1 - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
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
    return base64.b64encode(encoded_img).decode("utf-8")

def decode_from_base64(base64_str: str) -> UInt8Array:
    """Decode a base64 string to an image."""
    try:
        decoded_data = base64.b64decode(base64_str)
        np_arr = np.frombuffer(decoded_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        return img
    except Exception as e:
        raise ImageEncodingError(f"Failed to decode image: {e}")
