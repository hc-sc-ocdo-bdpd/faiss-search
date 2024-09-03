import numpy as np

class DatasetVariability:
    def __init__(self, embedding: np.ndarray, normalize: bool = True) -> None:
        if normalize:
            self.embedding = self.normalize(embedding)
        else:
            self.embedding = embedding

    def normalize(self, embedding: np.ndarray) -> None:
        return embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

    def variance(self) -> float:
        return np.sum(np.var(self.embedding, axis=0))
    
    def cosine_similarity_avg(self) -> float:
        mean_vec = np.mean(self.embedding, axis=0)
        return np.mean(np.dot(self.embedding, mean_vec))