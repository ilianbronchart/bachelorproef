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
git clone https://github.com/facebookresearch/sam2.git  libs/sam2
git clone https://github.com/mit-han-lab/efficientvit.git libs/efficientvit
git clone https://github.com/facebookresearch/dinov2.git libs/dinov2
poetry install
poetry run pip install -e libs/sam2/
poetry run pip install -e libs/groundingdino/
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