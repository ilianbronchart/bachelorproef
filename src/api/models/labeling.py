from dataclasses import dataclass


@dataclass
class PointAnnotation:
    label: str
    points: list[tuple[int, int]]
    point_labels: list[int]


@dataclass
class PointSegmentationRequest:
    annotations: list[PointAnnotation]
    frame: str


@dataclass
class AnnotationSaveRequest:
    frame_id: int
    annotations: list[dict]
