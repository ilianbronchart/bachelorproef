import time
from collections.abc import Generator
from pathlib import Path

import faiss
import numpy as np
import torch
import torchvision.transforms as T
from src.aliases import UInt8Array
from transformers import AutoImageProcessor, AutoModel, BitImageProcessor

IMAGE_PROCESSOR: BitImageProcessor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-base"
)
crop_size = IMAGE_PROCESSOR.crop_size["height"]
transformation_chain = T.Compose([
    T.ToTensor(),  # Convert numpy array (HWC) to tensor (CHW) scaled to [0,1]
    T.Resize(int((256 / 224) * crop_size)),
    T.CenterCrop(crop_size),
    T.Normalize(mean=IMAGE_PROCESSOR.image_mean, std=IMAGE_PROCESSOR.image_std),
])

EMBEDDING_DIM = 768


def load_model(device: str = "cuda") -> tuple[torch.nn.Module, BitImageProcessor]:
    return AutoModel.from_pretrained("facebook/dinov2-base").to(device)


def get_embeddings(
    dinov2: torch.nn.Module,
    samples: list[UInt8Array],
    batch_size: int = 64,
    log_performance: bool = False,
) -> Generator[tuple[torch.Tensor, int, int], None, None]:
    """
    Generate embeddings for a list of image samples using the DINOv2 model in batches.

    Args:
        samples (list[UInt8Array]): A list of image samples to process.
        batch_size (int, optional): The number of samples to process in each batch. Default is 64.

    Yields:
        tuple[torch.Tensor, int, int]:
            - torch.Tensor: The embeddings for the current batch, extracted from the model's last hidden state
              (using the first token, typically the [CLS] token).
            - int: The starting index of the current batch within the original samples list.
            - int: The ending index (exclusive) of the current batch within the original samples list.
    """
    samples_stacked = torch.stack([
        transformation_chain(sample).squeeze(0) for sample in samples
    ]).cuda()
    samples_batched = samples_stacked.split(batch_size, dim=0)

    start_time = time.time()
    with torch.no_grad():
        batch_start = 0
        for batch in samples_batched:
            embeddings = dinov2(batch).last_hidden_state[:, 0]
            current_batch_size = len(embeddings)
            yield embeddings, batch_start, batch_start + current_batch_size
            batch_start += current_batch_size

    # TODO: use logger
    if log_performance:
        sps = len(samples) / (time.time() - start_time)
        print(f"Generated {len(samples)} embeddings at {sps:.2f} samples per second")


def create_index() -> faiss.IndexIDMap:
    return faiss.IndexIDMap(faiss.IndexFlatL2(EMBEDDING_DIM))


def read_index(index_path: Path) -> faiss.IndexIDMap:
    return faiss.read_index(str(index_path))


def write_index(index: faiss.IndexIDMap, index_path: Path):
    faiss.write_index(index, str(index_path))


def search_index(index: faiss.IndexIDMap, embeddings: torch.Tensor, k: int = 100):
    vectors = embeddings.detach().cpu().numpy()
    vectors = np.float32(vectors)
    faiss.normalize_L2(vectors)
    distances, indices = index.search(vectors, k)
    return distances, indices


def add_embeddings_to_index(
    index: faiss.IndexIDMap, embeddings: torch.Tensor, sample_class_ids: UInt8Array
):
    vectors = embeddings.detach().cpu().numpy()
    vectors = np.float32(vectors)
    faiss.normalize_L2(vectors)
    index.add_with_ids(vectors, sample_class_ids)
