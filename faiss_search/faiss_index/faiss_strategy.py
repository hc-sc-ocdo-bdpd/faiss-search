import faiss
import numpy as np
from file_processing.tools.errors import UnsupportedHyperparameterError
from abc import ABC, abstractmethod


class FAISSStrategy(ABC):
    METRICS = {
        "L2": 1,
        "IP": 0,
    }

    def __init__(self, *args, metric: str=None, index=None):
        if index is not None:
            self.index = index
        else:
            try:
                metric_id = self.METRICS[metric]
            except KeyError:
                metric_id = 1
            self.index = self._create_index(*args, metric_id)

    def save_index(self, output_path: str):
        faiss.write_index(self.index, output_path)

    @abstractmethod
    def _create_index(self):
        pass

    @abstractmethod
    def query(self, xq: np.ndarray, k: int):
        if k < 1:
            raise UnsupportedHyperparameterError("k cannot be less than 1")
        return self.index.search(xq, k)