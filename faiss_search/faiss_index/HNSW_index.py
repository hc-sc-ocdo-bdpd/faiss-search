import faiss
import numpy as np
from faiss_search.faiss_index.faiss_strategy import FAISSStrategy


class HNSWIndex(FAISSStrategy):
    def _create_index(self, embeddings: np.ndarray, M: int,
                      efConstruction: int, metric: int):
        if M is None:
            M = 64
        if efConstruction is None:
            efConstruction = 64
        if not isinstance(M, int):
            raise TypeError("M must be an int type")
        if not isinstance(efConstruction, int):
            raise TypeError("efConstruction must be an int type")
        if M < 1:
            raise ValueError(
                "M cannot be less than 1")
        if efConstruction < 1:
            raise ValueError(
                "efConstruction cannot be less than 1")
        dimension = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dimension, M, metric)
        index.hnsw.efConstruction = efConstruction
        index.add(embeddings)
        return index

    def query(self, xq: np.ndarray, k: int = 1, efSearch: int = None):
        if efSearch is not None:
            if efSearch < 1:
                raise ValueError(
                    "efSearch cannot be less than 1")
            self.index.hnsw.efSearch = efSearch
        return super().query(xq, k)