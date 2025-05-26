from ultralytics import YOLO
import gc
import torch
from tqdm import tqdm
from experiment.settings import TRAINING_DATASETS_PATH, OBJECT_DETECTION_MODELS_PATH
from multiprocessing import Process
import shutil

OBJECT_DETECTION_DATASETS_PATH = TRAINING_DATASETS_PATH / "object_detection"

TRAIN_EPOCHS = 1
BATCH_SIZE = 32

datasets = list(OBJECT_DETECTION_DATASETS_PATH.iterdir())
if len(datasets) == 0:
    raise ValueError("No object detection datasets found in the specified path.")

print(f"Found {len(datasets)} object detection datasets.")

if OBJECT_DETECTION_MODELS_PATH.exists():
    shutil.rmtree(OBJECT_DETECTION_MODELS_PATH)
OBJECT_DETECTION_MODELS_PATH.mkdir(parents=True, exist_ok=True)


def train_model(dataset_path, model_name, crop_size):
    model = YOLO("yolo11n.pt")

    model.train(
        data=dataset_path / "data.yaml",
        epochs=TRAIN_EPOCHS,
        imgsz=int(crop_size),
        device="cuda",
        batch=BATCH_SIZE,
    )

    model_path = OBJECT_DETECTION_MODELS_PATH / f"{model_name}.pt"
    model.save(str(model_path))


for dataset_path in tqdm(datasets, desc="Training models"):
    model_name = dataset_path.name
    crop_size = model_name.split("_")[2]

    # run the training in a separate process (to avoid memory issues)
    print(f"Starting training for {model_name} with crop size {crop_size}")
    process = Process(target=train_model, args=(dataset_path, model_name, crop_size))
    process.start()
    process.join()
    print(f"Trained model for {model_name} with crop size {crop_size}")

    # Post-process cleanup
    gc.collect()
    torch.cuda.empty_cache()
    del process
