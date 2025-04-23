import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch


class FAISSIndexWithMetadata:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
        self.metadata = []

    def add(self, embeddings: torch.Tensor, metadata: list[dict[str, Any]]):
        if embeddings.shape[0] != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")

        # check if metadata values are all json serializable by trying to dump them
        for entry in metadata:
            try:
                json.dumps(entry)
            except TypeError:
                raise ValueError(f"Metadata entry {entry} is not JSON serializable")

        new_vectors_count = embeddings.shape[0]
        current_count = self.index.ntotal
        new_vector_ids = np.arange(current_count, current_count + new_vectors_count)

        vectors = embeddings.detach().cpu().numpy()
        vectors = np.float32(vectors)
        faiss.normalize_L2(vectors)
        self.index.add_with_ids(vectors, new_vector_ids)

        self.metadata.extend(metadata)

    def write(self, index_path: Path):
        faiss.write_index(self.index, str(index_path))
        metadata_path = Path(index_path).parent / (index_path.stem + ".json")
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f)

    @classmethod
    def load(cls, index_path: Path):
        index: faiss.IndexIDMap = faiss.read_index(str(index_path))
        metadata_path = Path(index_path).parent / (index_path.stem + ".json")
        with open(metadata_path) as f:
            metadata = json.load(f)
        instance = cls(index.d)
        instance.index = index
        instance.metadata = metadata
        return instance

    def search(self, embeddings: torch.Tensor, k: int):
        vectors = embeddings.detach().cpu().numpy()
        vectors = np.float32(vectors)
        faiss.normalize_L2(vectors)
        distances, indices = self.index.search(vectors, k)

        return distances, indices

    def get_metadata(self, i: int):
        if i < 0 or i >= self.index.ntotal:
            raise IndexError("Index out of bounds")
        return self.metadata[i]

    def get_metadatas(self, indices: list[int]):
        metadatas = []
        for i in indices:
            if i < 0 or i >= self.index.ntotal:
                raise IndexError("Index out of bounds")
            metadatas.append(self.metadata[i])
        return metadatas
