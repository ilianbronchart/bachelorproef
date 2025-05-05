import time
from collections.abc import Generator

import torch
import torchvision.transforms as T
from transformers import AutoImageProcessor, AutoModel, BitImageProcessor

from src.aliases import UInt8Array

IMAGE_PROCESSOR: BitImageProcessor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-base"
)
EMBEDDING_DIM = 768

crop_size = IMAGE_PROCESSOR.crop_size["height"]
transformation_chain = T.Compose([
    T.ToTensor(),  # Convert numpy array (HWC) to tensor (CHW) scaled to [0,1]
    T.Resize(int((256 / 224) * crop_size)),
    T.CenterCrop(crop_size),
    T.Normalize(mean=IMAGE_PROCESSOR.image_mean, std=IMAGE_PROCESSOR.image_std),
])


def load_model(device: str = "cuda") -> tuple[torch.nn.Module, BitImageProcessor]:
    return AutoModel.from_pretrained("facebook/dinov2-base").to(device)  # type: ignore[no-any-return]


def get_embeddings(
    dinov2: torch.nn.Module,
    samples: list[UInt8Array],
    batch_size: int = 64,
    log_performance: bool = False,
) -> Generator[tuple[torch.Tensor, int, int], None, None]:
    """
    Generate embeddings for a list of image samples
    using the DINOv2 model in batches.
    Processes data batch by batch so that not all
    samples are loaded into memory simultaneously.

    Args:
        samples (list[UInt8Array]): A list of image samples to process.
        batch_size (int, optional): The number of samples to process per batch.

    Yields:
        tuple[torch.Tensor, int, int]:
            - torch.Tensor: The embeddings for the current batch (from the [CLS] token).
            - int: The starting index of the current batch in the overall samples list.
            - int: The ending index (exclusive) of the current batch.
    """
    total_samples = len(samples)
    batch_start_index = 0
    start_time = time.time()

    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            # Process only the current batch
            current_batch_samples = samples[i : i + batch_size]
            # Apply your transformation chain for this batch of samples
            batch_tensor = torch.stack([
                transformation_chain(sample).squeeze(0)
                for sample in current_batch_samples
            ]).cuda()

            # Generate embeddings
            embeddings = dinov2(batch_tensor).last_hidden_state[:, 0]
            current_batch_size = embeddings.shape[0]

            yield embeddings, batch_start_index, batch_start_index + current_batch_size
            batch_start_index += current_batch_size

    if log_performance:
        sps = total_samples / (time.time() - start_time)
        print(f"Generated {total_samples} embeddings at {sps:.2f} samples per second")
