# bachelorproef

Trello board can be found [here](https://trello.com/b/iolCyuV2/bachelorproef)
Github repository can be found [here](https://github.com/ilianbronchart/bachelorproef)
Glasses 3 API Documentation can be found [here](https://tobiipro.github.io/g3pylib/g3pylib.html)

## Installing

### Just

```bash
sudo apt install just
```

### Docker

### Cuda

install tensorrt (see nvidia)

### Python Requirements

```bash
mkdir libs 
git clone https://github.com/tobiipro/g3pylib.git libs/g3pylib
git clone https://github.com/ilianbronchart/sam2.git libs/sam2
git clone https://github.com/facebookresearch/dinov2.git libs/dinov2
poetry install
poetry run pip install -e libs/sam2/
poetry run pip install git+https://github.com/DiGyt/cateyes.git
poetry run pip install -r libs/efficientvit/requirements.txt
poetry run python3 -m pip install --upgrade tensorrt tensorrt_bindings
poetry run pip  install tensorrt_libs --index-url https://pypi.nvidia.com
```

sudo cp libs/sam2/sam2/configs/sam2.1*.yaml libs/sam2/sam2/
```

### Model Checkpoints

```bash
just download-checkpoints
```

### Fix tensorrt:

```bash
cd .venv/lib/python3.10/site-packages/tensorrt
ln -s /usr/lib/x86_64-linux-gnu/libnvinfer.so.10 libnvinfer.so.8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.venv/lib/python3.10/site-packages/tensorrt
```

## Running the application in docker:

Install the nvidia container toolkit on the host machine:
```
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Download the tensorrt tar file by following the instructions [here](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html#download)
Put the tar file in the libs folder.

Run the following command to build the docker image:
```bash
just build
```

Run the following command to start the application in docker:
```bash
docker run eyetracking-app
```

Run the following command to start the application using the local src file:
```bash
docker run --rm -v "$(pwd)/src:/app/src" eyetracking-app
```

You can also run the application using an existing database file:
```bash
docker run --rm -v "$(pwd)/database.db:/app/database.db" eyetracking-app
```


