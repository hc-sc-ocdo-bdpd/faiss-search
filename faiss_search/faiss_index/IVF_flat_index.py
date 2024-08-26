import faiss
import numpy as np
from faiss_search.faiss_index.faiss_strategy import FAISSStrategy


class IVFFlatIndex(FAISSStrategy):
    def _create_index(self, embeddings: np.ndarray, nlist: int, metric: int):
        dimension = embeddings.shape[1]
        if nlist is None:
            nlist = max(1, int(np.sqrt(embeddings.shape[0] / 2)))
        if not isinstance(nlist, int):
            raise TypeError("nlist must be an int type")
        if nlist < 1:
            raise ValueError("nlist cannot be less than 1")
        if nlist > embeddings.shape[0]:
            raise ValueError(
                f"nlist value of {nlist} is larger than the number of documents in the index")
        quantizer = faiss.IndexFlat(dimension, metric)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, metric)
        index.train(embeddings)
        index.add(embeddings)
        return index

    def query(self, xq: np.ndarray, k: int = 1, nprobe: int = None):
        if nprobe is not None:
            if nprobe not in range(1, self.index.nlist + 1):
                raise ValueError(
                    f"nprobe must be between 1 and {self.index.nlist}")
            self.index.nprobe = nprobe
        return super().query(xq, k)