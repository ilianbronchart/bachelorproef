TENSORRT_VERSION := "10.9.0.34"
CUDA_USER_VERSION := "12.8"

lint:
    poetry run ruff format
    poetry run ruff check --fix src
    poetry run mypy src --strict
    
download-sam2-checkpoints:
    cd checkpoints && ../libs/sam2/checkpoints/download_ckpts.sh

download-groundingdino-checkpoints:
    cd checkpoints && wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

tensorrt:
    poetry run python scripts/export_fastsam_tensorrt.py

build:
    docker build \
      --build-arg TENSORRT_VERSION={{TENSORRT_VERSION}} \
      --build-arg CUDA_USER_VERSION={{CUDA_USER_VERSION}} \
      --tag eyetracking-app . \
      --progress=plain 