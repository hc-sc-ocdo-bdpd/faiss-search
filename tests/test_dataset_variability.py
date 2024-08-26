import os
import pytest
import numpy as np
from faiss_search import DatasetVariability

@pytest.fixture(scope="module")
def embeddings():
    return np.load("tests/resources/sample_embedding_files/paraphrasemini-space-embeddings.npy")

def test_normalization(embeddings):
    expected_result = np.ones(embeddings.shape[0])
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, expected_result, atol=1e-5)