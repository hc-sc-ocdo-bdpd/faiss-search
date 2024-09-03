import pytest
import numpy as np
from faiss_search import DatasetVariability

@pytest.fixture(scope="module")
def embeddings():
    return np.load("tests/resources/sample_embedding_files/paraphrasemini-space-embeddings.npy")

def test_normalization(embeddings):
    variability = DatasetVariability(embeddings)
    norms = np.linalg.norm(variability.embedding, axis=1)
    expected_result = np.ones(embeddings.shape[0])
    assert np.allclose(norms, expected_result, atol=1e-5)

def test_no_normalization(embeddings):
    variability = DatasetVariability(embeddings, normalize=False)
    norms = np.linalg.norm(variability.embedding, axis=1)
    normalized_result = np.ones(embeddings.shape[0])
    assert not np.allclose(norms, normalized_result, atol=1e-5)

@pytest.fixture(scope="module")
def variability(embeddings):
    return DatasetVariability(embeddings)

def test_cosine_similarity_in_range(variability):
    sim = variability.cosine_similarity_avg()
    assert (sim >= 0) and (sim <= 1)

def test_variance_in_range(variability):
    sim = variability.variance()
    assert (sim >= 0) and (sim <= 1)
