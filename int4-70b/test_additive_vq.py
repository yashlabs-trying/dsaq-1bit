import itertools
import unittest

import torch

from additive_vq import (_assign_beam2, _update_codebook, _metric_kmeans, _assign, block_ldlq,
                         hadamard_right, transform_hessian)


def weighted_error(x, centers, assignments, h_blocks, block_pos,
                   sample_weight=None):
    residual = x - centers[assignments]
    if sample_weight is None:
        sample_weight = torch.ones(x.shape[0])
    total = torch.tensor(0.0)
    for b in range(h_blocks.shape[0]):
        mask = block_pos == b
        r = residual[mask]
        total += ((r * (r @ h_blocks[b])).sum(1) * sample_weight[mask]).sum()
    return total


class HessianAdditiveVQTests(unittest.TestCase):
    def test_batched_metric_assignment_matches_direct_distances(self):
        torch.manual_seed(15)
        x = torch.randn(51, 4)
        centers = torch.randn(9, 4)
        positions = torch.arange(51) % 3
        a = torch.randn(3, 4, 4)
        h = a.transpose(1, 2) @ a + .1 * torch.eye(4)
        assigned = _assign(x, centers, h, positions)
        residual = x[:, None, :] - centers[None, :, :]
        distances = torch.einsum('nki,nij,nkj->nk', residual,
                                 h[positions], residual)
        torch.testing.assert_close(assigned, distances.argmin(1))

    def test_metric_restarts_keep_best_objective(self):
        torch.manual_seed(5)
        x = torch.randn(80, 4)
        positions = torch.arange(80) % 2
        a = torch.randn(2, 4, 4)
        h = a.transpose(1, 2) @ a + .1 * torch.eye(4)
        first = _metric_kmeans(x, 8, 4, 3, h, positions, restarts=1)
        best = _metric_kmeans(x, 8, 4, 3, h, positions, restarts=2)
        first_error = weighted_error(x, first, _assign(x, first, h, positions), h, positions)
        best_error = weighted_error(x, best, _assign(x, best, h, positions), h, positions)
        self.assertLessEqual(best_error, first_error + 1e-4)

    def test_randomized_hadamard_round_trip_and_metric(self):
        torch.manual_seed(4)
        x = torch.randn(3, 16)
        signs = torch.randint(0, 2, (16,)).float().mul_(2).sub_(1)
        transformed = hadamard_right(x, signs, block_size=8)
        restored = hadamard_right(
            transformed, signs, block_size=8, inverse=True)
        torch.testing.assert_close(restored, x, atol=2e-6, rtol=2e-6)

        a = torch.randn(16, 16)
        h = a.t() @ a + 0.1 * torch.eye(16)
        transformed_h = transform_hessian(h, signs, block_size=8)
        original_metric = torch.einsum('bi,ij,bj->b', x, h, x)
        transformed_metric = torch.einsum(
            'bi,ij,bj->b', transformed, transformed_h, transformed)
        torch.testing.assert_close(
            transformed_metric, original_metric, atol=2e-4, rtol=2e-5)

    def test_block_ldlq_returns_selected_codewords(self):
        torch.manual_seed(3)
        weights = torch.randn(6, 8)
        centers = torch.randn(16, 4)
        activations = torch.randn(40, 8)
        hessian = activations.t() @ activations + 0.1 * torch.eye(8)

        quantized, indices = block_ldlq(
            weights, hessian, centers, buf_cols=8)

        self.assertTrue(torch.isfinite(quantized).all())
        self.assertEqual(indices.shape, (6, 2))
        expected = centers[indices].reshape_as(weights)
        torch.testing.assert_close(quantized, expected)

    def test_weighted_update_reduces_weighted_error(self):
        torch.manual_seed(7)
        x = torch.randn(30, 3)
        assignments = torch.randint(0, 4, (30,))
        block_pos = torch.arange(30) % 2
        a = torch.randn(2, 3, 3)
        h_blocks = a.transpose(1, 2) @ a + 0.1 * torch.eye(3)
        old = torch.randn(4, 3)
        sample_weight = torch.rand(30).add_(0.1)

        updated = _update_codebook(
            x, assignments, 4, old, h_blocks, block_pos, sample_weight)

        self.assertLessEqual(
            weighted_error(x, updated, assignments, h_blocks, block_pos,
                           sample_weight),
            weighted_error(x, old, assignments, h_blocks, block_pos,
                           sample_weight) + 1e-4)

    def test_hessian_beam_matches_brute_force_when_beam_is_full(self):
        torch.manual_seed(11)
        x = torch.randn(9, 3)
        c0 = torch.randn(5, 3)
        c1 = torch.randn(4, 3)
        block_pos = torch.arange(9) % 2
        a = torch.randn(2, 3, 3)
        h_blocks = a.transpose(1, 2) @ a + 0.1 * torch.eye(3)

        i0, i1 = _assign_beam2(
            x, c0, c1, beam=len(c0), H_blocks=h_blocks,
            block_pos=block_pos)

        for row in range(x.shape[0]):
            h = h_blocks[block_pos[row]]
            errors = []
            for j, k in itertools.product(range(len(c0)), range(len(c1))):
                r = x[row] - c0[j] - c1[k]
                errors.append((float(r @ h @ r), j, k))
            _, expected0, expected1 = min(errors)
            self.assertEqual((int(i0[row]), int(i1[row])),
                             (expected0, expected1))


if __name__ == '__main__':
    unittest.main()
