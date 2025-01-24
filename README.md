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

Set CUDA_HOME (for grounding dino)
```bash
export CUDA_HOME=/usr/local/cuda-12.1/
```


### Python Requirements

```bash
mkdir libs 
git clone https://github.com/tobiipro/g3pylib.git libs/g3pylib
git clone https://github.com/facebookresearch/sam2.git  libs/sam2
git clone https://github.com/IDEA-Research/GroundingDINO.git libs/groundingdino
poetry install
poetry run pip install -e libs/sam2/
poetry run pip install -e libs/groundingdino/
sudo cp libs/sam2/sam2/configs/sam2.1*.yaml libs/sam2/sam2/
```

### Model Checkpoints

```bash
just download-checkpoints
```