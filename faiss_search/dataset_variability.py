import numpy as np

class DatasetVariability:
    def __init__(self, embedding: np.ndarray):
        self.embedding = embedding

    def variance(self) -> float:
        return np.sum(np.var(self.embedding, axis=0))
    
    def cosine_similarity_avg(self) -> float:
        mean_vec = np.mean(self.embedding, axis=0)
        return np.mean(np.dot(self.embedding, mean_vec))