TENSORRT_VERSION := "10.9.0.34"
CUDA_USER_VERSION := "12.8"

lint:
    poetry run ruff format
    poetry run ruff check --fix src
    poetry run mypy src --strict
    
build:
    docker build \
      --build-arg TENSORRT_VERSION={{TENSORRT_VERSION}} \
      --build-arg CUDA_USER_VERSION={{CUDA_USER_VERSION}} \
      --tag eyetracking-app . \
      --progress=plain

install:
    rm -rf libs
    mkdir libs

    git clone https://github.com/tobiipro/g3pylib.git libs/g3pylib
    git clone https://github.com/ilianbronchart/sam2.git libs/sam2
    # Copy the sam2 configs to the sam2 directory
    cp libs/sam2/sam2/configs/sam2.1/*.yaml libs/sam2/sam2/
    git clone https://github.com/facebookresearch/dinov2.git libs/dinov2

    poetry install
    poetry run pip install -e libs/sam2/

    # Download SAM2 checkpoints
    rm -rf checkpoints
    mkdir checkpoints
    cd checkpoints && ../libs/sam2/checkpoints/download_ckpts.sh
    cd ..

    # Download FastSAM checkpoints