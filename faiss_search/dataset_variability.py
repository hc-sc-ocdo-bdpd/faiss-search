import numpy as np

class DatasetVariability:
    def __init__(self, embedding: np.ndarray, normalize: bool = True) -> None:
        """
        Initializes the DatasetVariability class with the provided embedding.

        :param embedding: A 2D numpy array where each row is a vector representing an embedding.
        :param normalize: A boolean indicating whether to normalize the embeddings. Defaults to True.
        """
        if normalize:
            self.embedding = self.normalize(embedding)
        else:
            self.embedding = embedding

    def normalize(self, embedding: np.ndarray) -> None:
        """
        Normalizes the embeddings to have unit norm along the rows.

        :param embedding: A 2D numpy array where each row is a vector representing an embedding.

        :return: A 2D numpy array with normalized embeddings.
        """
        return embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

    def variance(self) -> float:
        """
        Computes the sum of the variance of the embeddings across features.
        Also equivalent to the average squared L2 distance between embeddings.

        :return: A float representing the sum of variances across features.
        """
        return np.sum(np.var(self.embedding, axis=0))
    
    def cosine_similarity_avg(self) -> float:
        """
        Calculates the average cosine similarity between embeddings.

        :return: A float representing the average cosine similarity of the embeddings.
        """
        mean_vec = np.mean(self.embedding, axis=0)
        return np.mean(np.dot(self.embedding, mean_vec))
