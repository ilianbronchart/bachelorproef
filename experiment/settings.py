from pathlib import Path

from src.config import UNKNOWN_CLASS_ID

TRIAL_RECORDING_IDS = [
    "67b71a70-da64-467a-9fb6-91bc29265fd1",
    "32f02db7-adc0-4556-a2da-ed2ba60a58c9",
    "b8eeecc0-06b1-47f7-acb5-89aab3c1724d",
    "d50c5f3b-2822-4462-9880-5a8f0dd46bfb",
    "9fa3e3b8-ed94-4b06-ba49-e66e3997d710",
    "98128cdc-ffeb-40cb-9528-573e25028e87",
    "89b60530-e0e4-4f5d-9ee6-af85c8d99ff4",
    "2fe01600-c057-40ee-8434-4e9e0688ca2d",
    "67823ccd-a1f0-4cde-b954-3b9e5fe160c1",
    "b214c60b-7521-495b-a699-e223da0c77c1",
    "b8f453aa-5a12-4cbb-a0ec-20eb503f8797",
    "7ae61789-7a26-4c31-abef-4ab49a34abfd",
    "6f3e2ccf-51f6-4377-8b84-63a3c16928a8",
    "5235be94-da01-43b5-8827-92a51d32ce30",
]

RECORDING_ID_TO_EXPERIMENT_NUMBER = {
    "67b71a70-da64-467a-9fb6-91bc29265fd1": 1,
    "32f02db7-adc0-4556-a2da-ed2ba60a58c9": 2,
    "b8eeecc0-06b1-47f7-acb5-89aab3c1724d": 3,
    "d50c5f3b-2822-4462-9880-5a8f0dd46bfb": 4,
    "9fa3e3b8-ed94-4b06-ba49-e66e3997d710": 5,
    "98128cdc-ffeb-40cb-9528-573e25028e87": 6,
    "89b60530-e0e4-4f5d-9ee6-af85c8d99ff4": 7,
    "2fe01600-c057-40ee-8434-4e9e0688ca2d": 8,
    "67823ccd-a1f0-4cde-b954-3b9e5fe160c1": 9,
    "b214c60b-7521-495b-a699-e223da0c77c1": 10,
    "b8f453aa-5a12-4cbb-a0ec-20eb503f8797": 11,
    "7ae61789-7a26-4c31-abef-4ab49a34abfd": 12,
    "6f3e2ccf-51f6-4377-8b84-63a3c16928a8": 13,
    "5235be94-da01-43b5-8827-92a51d32ce30": 12,
}

RECORDING_ID_TO_EXPERIMENT_OBJECTS = {
    "67b71a70-da64-467a-9fb6-91bc29265fd1": [
        "naaldcontainer",
        "spuit",
        "keukenmes",
        "infuus",
        "stethoscoop",
    ],
    "32f02db7-adc0-4556-a2da-ed2ba60a58c9": [
        "bol wol",
        "snoep",
        "nuchter",
        "fotokader",
        "iced tea",
    ],
    "b8eeecc0-06b1-47f7-acb5-89aab3c1724d": [
        "bril",
        "monitor",
        "rollator",
        "ampulevloeistof",
        "ampulepoeder",
    ],
    "d50c5f3b-2822-4462-9880-5a8f0dd46bfb": [
        "snoep",
        "naaldcontainer",
        "infuus",
        "bol wol",
        "rollator",
    ],
    "9fa3e3b8-ed94-4b06-ba49-e66e3997d710": [
        "iced tea",
        "stethoscoop",
        "keukenmes",
        "fotokader",
        "spuit",
    ],
    "98128cdc-ffeb-40cb-9528-573e25028e87": [
        "monitor",
        "ampulevloeistof",
        "ampulepoeder",
        "nuchter",
        "bril",
    ],
    "89b60530-e0e4-4f5d-9ee6-af85c8d99ff4": [
        "nuchter",
        "stethoscoop",
        "monitor",
        "spuit",
        "ampulevloeistof",
    ],
    "2fe01600-c057-40ee-8434-4e9e0688ca2d": [
        "naaldcontainer",
        "ampulepoeder",
        "bol wol",
        "rollator",
        "infuus",
    ],
    "67823ccd-a1f0-4cde-b954-3b9e5fe160c1": [
        "keukenmes",
        "bril",
        "snoep",
        "iced tea",
        "fotokader",
    ],
    "b214c60b-7521-495b-a699-e223da0c77c1": [
        "naaldcontainer",
        "ampulepoeder",
        "iced tea",
        "monitor",
        "keukenmes",
    ],
    "b8f453aa-5a12-4cbb-a0ec-20eb503f8797": [
        "infuus",
        "nuchter",
        "spuit",
        "bril",
        "bol wol",
    ],
    "7ae61789-7a26-4c31-abef-4ab49a34abfd": [
        "rollator",
        "fotokader",
        "stethoscoop",
        "snoep",
        "ampulevloeistof",
    ],
    "6f3e2ccf-51f6-4377-8b84-63a3c16928a8": [
        "snoep",
        "bol wol",
        "infuus",
        "stethoscoop",
        "monitor",
    ],
    "5235be94-da01-43b5-8827-92a51d32ce30": [
        "rollator",
        "fotokader",
        "ampulevloeistof",
        "stethoscoop",
        "snoep",
    ],
}

LABELING_REC_SAME_BACKGROUND_ID = "d6fd0aed-b901-4863-bad8-7910dad693e0"
LABELING_REC_DIFF_BACKGROUND_ID = "73ce8a30-ccc6-4514-b978-f8b5844be16b"

FULLY_LABELED_RECORDINGS = [
    "67b71a70-da64-467a-9fb6-91bc29265fd1",
    "32f02db7-adc0-4556-a2da-ed2ba60a58c9",
    "b8eeecc0-06b1-47f7-acb5-89aab3c1724d",
    "d50c5f3b-2822-4462-9880-5a8f0dd46bfb",
    "9fa3e3b8-ed94-4b06-ba49-e66e3997d710",
    "98128cdc-ffeb-40cb-9528-573e25028e87",
    "89b60530-e0e4-4f5d-9ee6-af85c8d99ff4",
    "2fe01600-c057-40ee-8434-4e9e0688ca2d",
    "67823ccd-a1f0-4cde-b954-3b9e5fe160c1",
    "b8f453aa-5a12-4cbb-a0ec-20eb503f8797",
]

MISSING_PREDICTION_CLASS_ID = -2
MISSING_GROUND_TRUTH_CLASS_ID = -3
CLASS_ID_TO_NAME = {
    1: "naaldcontainer",
    2: "spuit",
    3: "keukenmes",
    4: "infuus",
    5: "stethoscoop",
    6: "bol wol",
    7: "snoep",
    8: "nuchter",
    9: "fotokader",
    10: "iced tea",
    11: "bril",
    12: "monitor",
    13: "rollator",
    14: "ampulevloeistof",
    15: "ampulepoeder",
    UNKNOWN_CLASS_ID: "unknown",
    MISSING_PREDICTION_CLASS_ID: "geen voorspelling",
    MISSING_GROUND_TRUTH_CLASS_ID: "geen grondwaarheid"
}
NAME_TO_CLASS_ID = {name: class_id for class_id, name in CLASS_ID_TO_NAME.items()}
IGNORED_CLASS_IDS = [
    NAME_TO_CLASS_ID["ampulepoeder"]
]

SORTED_CLASS_IDS = sorted(CLASS_ID_TO_NAME.keys())
CLASS_NAMES = sorted(CLASS_ID_TO_NAME.values())

# Data Paths
LABELING_VALIDATION_VIDEOS_PATH = Path("data/labeling_validation_videos")
GROUND_TRUTH_PATH = Path("data/ground_truth.csv")
VECTOR_INDEXES_PATH = Path("data/vector_indexes")
SAME_BACKGROUND_INDEXES_PATH = VECTOR_INDEXES_PATH / "same_background"
DIFF_BACKGROUND_INDEXES_PATH = VECTOR_INDEXES_PATH / "diff_background"
GAZE_SEGMENTATION_RESULTS_PATH = Path("data/gaze_segmentation_results")
OBJECT_DATASETS_PATH = Path("data/object_datasets")
FINAL_PREDICTIONS_PATH = Path("data/final_predictions")
OBJECT_DETECTION_PREDICTIONS_PATH = FINAL_PREDICTIONS_PATH / "object_detection"
FINAL_PREDICTION_VIDEOS_PATH = Path("data/final_prediction_videos")
TRAINING_DATASETS_PATH = Path("data/training_datasets")
RECORDING_FRAMES_PATH = Path("data/recording_frames")
YOLO_MODELS_PATH = Path("data/yolo_models")
RECORDINGS_PATH = Path("data/recordings")
PROCESSED_TRACKING_RESULTS_PATH = Path("data/processed_tracking_results")
BLUR_METRICS_PATH = Path("data/blur_metrics")
MODELS_PATH = Path("data/models")
OBJECT_DETECTION_MODELS_PATH = MODELS_PATH / "object_detection"
COMPARISON_SETS_PATH = Path("data/comparison_sets")

# Database
SIMROOM_ID = 1
