from ultralytics import FastSAM

model = FastSAM('checkpoints/FastSAM-x.pt')

for result in model.track(source='data/recordings/39f5164f-873d-4d6b-be6b-e1d5db79c02a.mp4', imgsz=640, save=True, show=True, stream=True):
    continue