from dataclasses import dataclass
from typing import List
from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
import base64
from src.api.settings import app

router = APIRouter(prefix="/inference")

@dataclass
class PointAnnotation:
    label: str
    points: List[tuple[int, int]]
    point_labels: List[int]

@dataclass
class PointSegmentationRequest:
    annotations: List[PointAnnotation]
    frame: str

@router.post("/", response_class=HTMLResponse)
async def create_inference(response: Response, model_name: str):
    print("Creating inference")
    app.is_inference_running = True
    return response
    
@router.post("/segmentation")
async def get_point_segmentation(request: Request, body: PointSegmentationRequest):
    print(await request.json())
    print(body)

    # data = await request.json()

    # # For now, generate a dummy mask for each class.
    # # This dummy mask is a 1x1 transparent PNG encoded in base64.
    # dummy_png_base64 = (
    #     "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAD"
    #     "hgH9Z0Z/gAAAAABJRU5ErkJggg=="
    # )

    # dummy_masks = []
    # for cls in classes:
    #     label = cls.get("label", "unknown")
    #     dummy_masks.append({
    #         "class": label,
    #         "mask": dummy_png_base64
    #     })

    # return JSONResponse(content=dummy_masks)
