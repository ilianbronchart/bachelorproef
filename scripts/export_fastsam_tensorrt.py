from src.core.config import CHECKPOINTS_PATH
from ultralytics import FastSAM

models = ["FastSAM-x.pt"]
sizes = [512]

for model_name in models:
    for size in sizes:
        model = FastSAM(CHECKPOINTS_PATH / model_name)
        model.export(
            format="engine",
            imgsz=size,
            batch=1,
            device="cuda",
            verbose=False,
            half=True,
        )
