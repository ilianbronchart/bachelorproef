from dataclasses import dataclass

import cv2
from fastapi import APIRouter, Depends, Response
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from src.api.dependencies import get_labeler, get_selected_class_id
from src.api.models import (
    Labeler,
    LabelingAnnotationsContext,
    LabelingClassesContext,
    LabelingControlsContext,
    Request,
)
from src.config import Template, templates
from src.db import CalibrationRecording, engine
from src.db.models import Annotation, PointLabel, SimRoomClass
from src.utils import encode_to_png, is_hx_request

router = APIRouter(prefix="/labeling")


@router.post("/", response_class=Response)
async def start_labeling(request: Request, calibration_id: int) -> Response:
    with Session(engine) as session:
        cal_rec = (
            session.query(CalibrationRecording)
            .filter(CalibrationRecording.id == calibration_id)
            .first()
        )

        if not cal_rec:
            return Response(
                status_code=404, content="Error: Calibration recording not found"
            )

        try:
            request.app.labeler = Labeler(calibration_recording=cal_rec)
            labeling_context = request.app.labeler.get_labeling_context(request).to_dict()

            if is_hx_request(request):
                return templates.TemplateResponse(Template.LABELER, labeling_context)
            return templates.TemplateResponse(Template.INDEX, labeling_context)
        except Exception as e:
            return Response(status_code=500, content=f"Failed to start labeling: {e!s}")


@router.get("/", response_class=HTMLResponse)
async def labeling(
    request: Request, labeler: Labeler = Depends(get_labeler)
) -> HTMLResponse:
    labeling_context = labeler.get_labeling_context(request).to_dict()
    if is_hx_request(request):
        return templates.TemplateResponse(Template.LABELER, labeling_context)
    return templates.TemplateResponse(Template.INDEX, labeling_context)


@router.get("/point_labels", response_class=JSONResponse)
async def point_labels(labeler: Labeler = Depends(get_labeler)) -> JSONResponse:
    with Session(engine) as session:
        cal_rec_id = labeler.calibration_recording.id
        annotations = (
            session.query(Annotation)
            .filter(
                Annotation.calibration_recording_id == cal_rec_id,
                Annotation.frame_idx == labeler.current_frame_idx,
            )
            .all()
        )

        point_labels = []
        for annotation in annotations:
            for point_label in annotation.point_labels:
                point_label_dict = point_label.to_dict()
                point_label_dict["class_id"] = annotation.sim_room_class_id
                point_labels.append(point_label_dict)

        return JSONResponse(content=point_labels)


@router.get("/current_frame", response_class=Response)
async def current_frame(labeler: Labeler = Depends(get_labeler)) -> Response:
    with Session(engine) as session:
        annotations = (
            session.query(Annotation)
            .filter(
                Annotation.calibration_recording_id == labeler.calibration_recording.id,
                Annotation.frame_idx == labeler.current_frame_idx,
            )
            .all()
        )

        frame = labeler.get_overlay(annotations)
        ret, encoded_png = cv2.imencode(".png", frame)
        if not ret:
            return Response(status_code=500, content="Error: Failed to encode frame")

        return Response(content=encoded_png.tobytes(), media_type="image/png")


@router.get("/controls", response_class=HTMLResponse)
async def controls(
    request: Request,
    frame_idx: int,
    labeler: Labeler = Depends(get_labeler),
    selected_class_id: int = Depends(get_selected_class_id),
) -> HTMLResponse:
    labeler.seek(frame_idx)
    context = LabelingControlsContext(
        request=request,
        current_frame_idx=labeler.current_frame_idx,
        frame_count=labeler.frame_count,
        selected_class_id=selected_class_id,
    )

    return templates.TemplateResponse(
        Template.LABELING_CONTROLS, context=context.to_dict()
    )


@router.get("/classes", response_class=HTMLResponse)
async def classes(
    request: Request, selected_class_id: int = -1, labeler: Labeler = Depends(get_labeler)
) -> HTMLResponse:
    with Session(engine) as session:
        classes = (
            session.query(SimRoomClass)
            .filter(SimRoomClass.sim_room_id == labeler.calibration_recording.sim_room_id)
            .all()
        )

        if selected_class_id == -1 and len(classes) > 0:
            selected_class_id = classes[0].id

        labeler.selected_class_id = selected_class_id

        context = LabelingClassesContext(
            request=request,
            selected_class_id=selected_class_id,
            sim_room_id=labeler.calibration_recording.sim_room_id,
            classes=classes,
        )

        return templates.TemplateResponse(
            Template.LABELING_CLASSES, context=context.to_dict()
        )


@dataclass
class AnnotationPostBody:
    point: tuple[int, int]
    label: int
    delete_point: bool = False


@router.get("/annotations", response_class=HTMLResponse)
async def annotations(
    request: Request,
    labeler: Labeler = Depends(get_labeler),
    selected_class_id: int = Depends(get_selected_class_id),
) -> HTMLResponse:
    context = LabelingAnnotationsContext(request=request)
    with Session(engine) as session:
        cal_rec_id = labeler.calibration_recording.id
        context.annotations = (
            session.query(Annotation)
            .filter(
                Annotation.calibration_recording_id == cal_rec_id,
                Annotation.sim_room_class_id == selected_class_id,
            )
            .all()
        )

    return templates.TemplateResponse(Template.LABELING_ANNOTATIONS, context.to_dict())


@router.post("/annotations", response_class=Response)
async def post_annotation(
    body: AnnotationPostBody,
    labeler: Labeler = Depends(get_labeler),
    selected_class_id: int = Depends(get_selected_class_id),
) -> Response:
    if selected_class_id == -1:
        return Response(status_code=400, content="Error: No class selected")

    with Session(engine) as session:
        annotation = (
            session.query(Annotation)
            .filter(
                Annotation.calibration_recording_id == labeler.calibration_recording.id,
                Annotation.frame_idx == labeler.current_frame_idx,
                Annotation.sim_room_class_id == selected_class_id,
            )
            .first()
        )

        if body.delete_point and annotation is None:
            return Response(status_code=404, content="Annotation not found")

        if not annotation:
            annotation = Annotation(
                calibration_recording_id=labeler.calibration_recording.id,
                sim_room_class_id=selected_class_id,
                frame_idx=labeler.current_frame_idx,
            )
            session.add(annotation)
            session.flush()

        if body.delete_point:
            closest_point = PointLabel.find_closest(
                annotation.id, body.point[0], body.point[1]
            )
            if not closest_point:
                return Response(status_code=404, content="Point not found")

            session.delete(closest_point)
        else:
            point_label = PointLabel(
                annotation_id=annotation.id,
                x=body.point[0],
                y=body.point[1],
                label=body.label,
            )
            session.add(point_label)

        session.flush()
        points = [(label.x, label.y) for label in annotation.point_labels]
        labels = [int(label.label) for label in annotation.point_labels]

        # delete annotation if no points remain
        if len(points) == 0:
            session.delete(annotation)
            session.commit()
            return await current_frame(labeler)

        # update annotation mask and bounding box
        result = labeler.predict_image(points, labels)
        if result is None:
            return Response(status_code=500, content="Error: Failed to predict image")

        annotation.annotation_mask = encode_to_png(result.mask)
        annotation.bounding_box = ",".join(map(str, result.bounding_box))
        annotation.frame_crop = encode_to_png(result.frame_crop)

        session.commit()
        return await current_frame(labeler)


@router.delete("/annotations/{annotation_id}", response_class=HTMLResponse)
async def delete_calibration_annotation(
    request: Request, annotation_id: int, labeler: Labeler = Depends(get_labeler)
) -> HTMLResponse:
    with Session(engine) as session:
        annotation = (
            session.query(Annotation).filter(Annotation.id == annotation_id).first()
        )

        if not annotation:
            return HTMLResponse(status_code=404, content="Annotation not found")

        sim_room_class_id = annotation.sim_room_class_id
        session.delete(annotation)
        session.commit()

        return await annotations(request, labeler, sim_room_class_id)


@router.get("/tracking/status", response_class=HTMLResponse)
async def tracking_status(
    request: Request, labeler: Labeler = Depends(get_labeler)
) -> HTMLResponse:
    if labeler.tracking_job is None:
        return HTMLResponse(status_code=204)

    return await controls(request, labeler.current_frame_idx, labeler)


# def save_segmentation(frame_idx: int, seg_data: dict, output_dir: str) -> None:
#     try:
#         # Define the output file path. Adjust the path as needed.
#         file_path = os.path.join(output_dir, f"{frame_idx:05d}.npz")
#         np.savez_compressed(file_path, **seg_data)
#         # Optionally, confirm the file was saved
#     except Exception as e:
#         logger.exception(f"Error saving segmentation for frame {frame_idx}: {e}")
#     finally:
#         # Free up memory.
#         del seg_data


# @router.post("/tracking", response_class=JSONResponse)
# async def tracking(labeler: Labeler = Depends(get_labeler)) -> JSONResponse:
#     with Session(engine) as session:
#         annotations = session.query(Annotation).filter(
#             Annotation.calibration_recording_id == labeler.calibration_recording.id
#         ).all()

#         video_predictor = load_sam2_video_predictor(Sam2Checkpoints.LARGE)
#         inference_state = video_predictor.init_state(video_path=str(FRAMES_PATH), async_loading_frames=True)

#         for annotation in annotations:
#             point_labels = annotation.point_labels
#             points = [(int(point_label.x), int(point_label.y)) for point_label in point_labels]
#             labels = [point_label.label for point_label in point_labels]

#             video_predictor.add_new_points(
#                 inference_state=inference_state,
#                 frame_idx=annotation.frame_idx,
#                 obj_id=annotation.sim_room_class.id,
#                 points=points,
#                 labels=labels,
#             )

#         # Define output directory for segmentation results.
#         output_dir = "./segmentation_results"
#         # empty the directory
#         if os.path.exists(output_dir):
#             for file in os.listdir(output_dir):
#                 os.remove(os.path.join(output_dir, file))
#         os.makedirs(output_dir, exist_ok=True)
#         logger.info(f"Segmentation output directory set to {output_dir}")

#         # Use ThreadPoolExecutor to offload saving to disk.r
#         futures = []
#         with ProcessPoolExecutor() as executor:
#             with torch.amp.autocast("cuda"):
#                 for out_frame_idx, class_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
#                     seg_data = {}
#                     for i, out_obj_id in enumerate(class_ids):
#                         mask = (out_mask_logits[i] > 0.5).detach()
#                         if mask.any():
#                             seg_data[str(out_obj_id)] = mask.cpu().numpy()

#                     future = executor.submit(save_segmentation, out_frame_idx, seg_data, output_dir)
#                     futures.append(future)

#     logger.info("Tracking complete")
#     return JSONResponse(content={"status": "tracking complete"})
