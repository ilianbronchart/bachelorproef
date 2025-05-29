from ultralytics import YOLO
import gc
import torch
from tqdm import tqdm
from experiment.settings import TRAINING_DATASETS_PATH, OBJECT_DETECTION_MODELS_PATH
from multiprocessing import Process

OBJECT_DETECTION_DATASETS_PATH = TRAINING_DATASETS_PATH / "object_detection"

TRAIN_EPOCHS = 100
BATCH_SIZE = 0.9
PATIENCE = 10

datasets = list(OBJECT_DETECTION_DATASETS_PATH.iterdir())
if len(datasets) == 0:
    raise ValueError("No object detection datasets found in the specified path.")


def train_model(dataset_path, model_name, crop_size):
    model = YOLO("yolo11l.pt")
    model.train(
        data=dataset_path / "data.yaml",
        epochs=TRAIN_EPOCHS,
        imgsz=int(crop_size),
        device="cuda",
        batch=BATCH_SIZE,
        patience=PATIENCE,
        plots=True,
        save=True,
    )

for dataset_path in tqdm(datasets, desc="Training models"):
    model_name = dataset_path.name
    crop_size = model_name.split("_")[2]
    sample_count = int(model_name.split("_")[-1])

    if sample_count != 2000:
        print(f"Skipping dataset {model_name} with sample count {sample_count}. Large version is only trained on 2000 samples.")
        continue


    # run the training in a separate process (to avoid memory issues)
    print("\n--------------------------------------------------------------")
    print(f"Starting training for {model_name} with crop size {crop_size}")
    print("--------------------------------------------------------------\n")
    process = Process(target=train_model, args=(dataset_path, model_name, crop_size))
    process.start()
    process.join()
    print(f"Trained model for {model_name} with crop size {crop_size}")

    # Post-process cleanup
    gc.collect()
    torch.cuda.empty_cache()
    del process
