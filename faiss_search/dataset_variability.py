import numpy as np

class DatasetVariability:
    def __init__(self, embedding: np.ndarray):
        self.embedding = embedding

    def variance(self):
        return np.sum(np.var(self.embedding, axis=0))