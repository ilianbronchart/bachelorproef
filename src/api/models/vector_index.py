import json
from pathlib import Path
from typing import Any, cast

import faiss
import numpy as np
import torch


class FAISSIndexWithMetadata:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
        self.metadata: list[dict[str, Any]] = []

    def add(self, embeddings: torch.Tensor, metadata: list[dict[str, Any]]) -> None:
        if embeddings.shape[0] != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")

        # check if metadata values are all json serializable by trying to dump them
        for entry in metadata:
            try:
                json.dumps(entry)
            except TypeError as e:
                raise ValueError(
                    f"Metadata entry {entry} is not JSON serializable"
                ) from e

        new_vectors_count = embeddings.shape[0]
        current_count = self.index.ntotal
        new_vector_ids = np.arange(current_count, current_count + new_vectors_count)

        vectors = embeddings.detach().cpu().numpy().astype(np.float32)
        faiss.normalize_L2(vectors)
        self.index.add_with_ids(vectors, new_vector_ids)

        self.metadata.extend(metadata)

    def write(self, index_path: Path) -> None:
        faiss.write_index(self.index, str(index_path))
        metadata_path = Path(index_path).parent / (index_path.stem + ".json")
        with Path.open(metadata_path, "w") as f:
            json.dump(self.metadata, f)

    @classmethod
    def load(cls, index_path: Path) -> "FAISSIndexWithMetadata":
        index: faiss.IndexIDMap = faiss.read_index(str(index_path))
        metadata_path = Path(index_path).parent / (index_path.stem + ".json")
        with Path.open(metadata_path) as f:
            metadata = json.load(f)
        instance = cls(index.d)
        instance.index = index
        instance.metadata = metadata
        return instance

    def search(
        self, embeddings: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vectors = embeddings.detach().cpu().numpy().astype(np.float32)
        faiss.normalize_L2(vectors)
        distances, indices = self.index.search(vectors, k)

        return cast(torch.Tensor, distances), cast(torch.Tensor, indices)

    def get_metadata(self, i: int) -> dict[str, Any]:
        if i < 0 or i >= self.index.ntotal:
            raise IndexError("Index out of bounds")
        return self.metadata[i]

    def get_metadatas(self, indices: list[int]) -> list[dict[str, Any]]:
        metadatas = []
        for i in indices:
            if i < 0 or i >= self.index.ntotal:
                raise IndexError("Index out of bounds")
            metadatas.append(self.metadata[i])
        return metadatas
