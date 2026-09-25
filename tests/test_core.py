import numpy as np
import torch

from dsaq.core import (
    blocks_to_matrix,
    candidate_metadata_bits,
    fit_router,
    hadamard_blocks,
    matrix_to_blocks,
    seeded_candidates,
)


def test_block_roundtrip_and_hadamard():
    x = torch.randn(9, 13)
    blocks, layout = matrix_to_blocks(x, 8)
    assert torch.allclose(blocks_to_matrix(blocks, layout), x)
    assert torch.allclose(hadamard_blocks(hadamard_blocks(blocks, 8), 8), blocks, atol=1e-5)


def test_seeded_candidates_are_reproducible():
    blocks = torch.randn(2, 64)
    valid = torch.ones_like(blocks, dtype=torch.bool)
    a = seeded_candidates(blocks, valid, [3, 7], "m", "gaussian", 4, 123)
    b = seeded_candidates(blocks, valid, [3, 7], "m", "gaussian", 4, 123)
    assert torch.equal(a, b)


def test_metadata_and_router():
    assert candidate_metadata_bits("gaussian", 8, 10) == 350
    rng = np.random.default_rng(3)
    x = rng.normal(size=(100, 3)); gain = x[:, 0] - .2 * x[:, 1]
    cal = np.arange(100) % 2 == 0
    model = fit_router(x, gain, cal, minimum_useful_gain=0.0)
    assert np.corrcoef(model["scores"][~cal], gain[~cal])[0, 1] > .95
