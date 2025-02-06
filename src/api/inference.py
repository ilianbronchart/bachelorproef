from dataclasses import dataclass

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse
from src.config import FAST_SAM_CHECKPOINT, Request
from src.core.utils import base64_to_numpy
from src.logic.inference import fastsam
from ultralytics import FastSAM

router = APIRouter(prefix="/inference")


@dataclass
class PointAnnotation:
    label: str
    points: list[tuple[int, int]]
    point_labels: list[int]


@dataclass
class PointSegmentationRequest:
    annotations: list[PointAnnotation]
    frame: str


@router.put("/FastSAM")
async def load_sam_model(request: Request):
    if request.app.is_inference_running and request.app.inference_model is not None:
        del request.app.inference_model

    request.app.inference_model = FastSAM(FAST_SAM_CHECKPOINT)
    request.app.is_inference_running = True
    return Response(content="Model loaded successfully", status_code=200)


@router.post("/FastSAM", response_class=JSONResponse)
async def get_point_segmentation(request: Request, body: PointSegmentationRequest):
    # read base64 encoded image from request to numpy array
    image = base64_to_numpy(body.frame)
    results = []
    print(body.frame[:100])

    for annotation in body.annotations:
        points = annotation.points
        point_labels = annotation.point_labels
        label = annotation.label

        # get masks for each annotation
        mask = fastsam.get_mask(
            image=image, model=request.app.inference_model, points=points, point_labels=point_labels, conf=0.6, iou=0.8
        )

        print(mask.shape)

        # convert torch tensor to base64 encoded str

        # results.append({
        #         "class": label,
        #         "mask": mask,
        #         "point_label": point_labels[i]
        #     })

    # dummy_masks = []
    # for cls in classes:
    #     label = cls.get("label", "unknown")
    #     dummy_masks.append({
    #         "class": label,
    #         "mask": dummy_png_base64
    #     })

    # return JSONResponse(content=dummy_masks)
