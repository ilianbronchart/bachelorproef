check:
    poetry run ruff format
    poetry run ruff check --fix
    poetry run mypy . 

run:
    poetry run fastapi dev src/main.py
    
download-sam2-checkpoints:
    cd checkpoints && ../libs/sam2/checkpoints/download_ckpts.sh

download-groundingdino-checkpoints:
    cd checkpoints && wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

tensorrt:
    poetry run python scripts/export_fastsam_tensorrt.py