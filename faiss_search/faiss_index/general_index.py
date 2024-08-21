import numpy as np
from faiss_search.faiss_index.faiss_strategy import FAISSStrategy


class GeneralIndex(FAISSStrategy):
    def _create_index(self):
        raise NotImplementedError()

    def query(self, xq: np.ndarray, k: int = 1):
        return super().query()