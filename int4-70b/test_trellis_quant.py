import tempfile
import unittest
import os
import subprocess
import sys
from pathlib import Path
import torch
from trellis_quant import (decode_paths, viterbi, fixed_nodes, save_layer, load_layer,
                           FORCED_BLOCKS, train_path_nodes, train_importance_nodes,
                           train_residual_specialists, learn_outlier_nodes,
                           metadata_allocation, METADATA_BYTES, PATH_BYTES)


class TrellisTests(unittest.TestCase):
    def test_global_metadata_allocation_pays_budget_and_protects_attention(self):
        names = [f'model.layers.{i}.{kind}.{projection}' for i in range(24)
                 for kind, projection in [('self_attn', 'q_proj'), ('self_attn', 'k_proj'),
                    ('self_attn', 'v_proj'), ('self_attn', 'o_proj'), ('mlp', 'gate_proj'),
                    ('mlp', 'up_proj'), ('mlp', 'down_proj')]]
        allocation = metadata_allocation(names, 'global_mlp')
        paid = sum(allocation.values()) * PATH_BYTES
        required = len(names) * METADATA_BYTES
        self.assertGreaterEqual(paid, required)
        self.assertLess(paid - required, PATH_BYTES)
        self.assertTrue(all(count == 0 for name, count in allocation.items() if 'self_attn' in name))
        self.assertTrue(all(count > 0 for name, count in allocation.items() if '.mlp.' in name))

    def test_uint64_paths_round_trip_including_sign_bit(self):
        script = '''
import tempfile
from pathlib import Path
import torch
from trellis_quant import save_layer, load_layer, fixed_nodes, FORCED_BLOCKS, decode_paths
ids = torch.tensor([-1, -(2**63), 12345]).repeat(100).reshape(100, 3)
ids.reshape(-1)[-FORCED_BLOCKS:] = 0
nodes = fixed_nodes('cpu')
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'layer.bin'
    save_layer(path, ids, nodes, 96, 17, .125)
    restored, restored_nodes, seed, scale = load_layer(path)
    torch.testing.assert_close(restored, ids, atol=0, rtol=0)
    torch.testing.assert_close(decode_paths(restored, restored_nodes), decode_paths(ids, nodes))
    assert path.stat().st_size * 8 / (100 * 96) <= 2
'''
        env = dict(os.environ, FLASHRUN_TRELLIS_WIDTH='32')
        subprocess.run([sys.executable, '-c', script], env=env, check=True)

    def test_full_metric_rerank_beats_single_diagonal_candidate(self):
        torch.manual_seed(27)
        nodes = fixed_nodes('cpu')
        x = torch.randn(5, 16)
        a = torch.randn(16, 16)
        h = a.t() @ a + torch.eye(16)
        basic, _ = viterbi(x, nodes, h.diag())
        reranked, paths = viterbi(x, nodes, h.diag(), full_metric=h)
        before = (((x - basic) @ h) * (x - basic)).sum(1)
        after = (((x - reranked) @ h) * (x - reranked)).sum(1)
        self.assertTrue(torch.all(after <= before + 1e-4))
        torch.testing.assert_close(reranked, decode_paths(paths, nodes, width=16))

    def test_batch_diagonal_matches_independent_viterbi(self):
        torch.manual_seed(31)
        nodes = fixed_nodes('cpu')
        x = torch.randn(4, 8)
        diagonal = torch.rand(4, 8) + .1
        batched, paths = viterbi(x, nodes, diagonal)
        separate = torch.cat([viterbi(x[i:i + 1], nodes, diagonal[i])[0]
                              for i in range(len(x))])
        torch.testing.assert_close(batched, separate)
        torch.testing.assert_close(batched, decode_paths(paths, nodes))

    def test_sixteen_weight_path_decode_and_optimality_lower_bound(self):
        torch.manual_seed(21)
        nodes = fixed_nodes('cpu')
        x = torch.randn(4, 16)
        q, paths = viterbi(x, nodes)
        torch.testing.assert_close(q, decode_paths(paths, nodes, width=16))
        # Optimal search must beat each of these independently drawn paths.
        random_ids = torch.randint(2**32, (500,))
        candidates = decode_paths(random_ids, nodes, width=16)
        sampled_best = (x[:, None] - candidates[None]).square().sum(-1).min(1).values
        self.assertTrue(torch.all((x - q).square().sum(1) <= sampled_best + 1e-5))

    def test_path_node_fitting_reduces_training_distortion(self):
        torch.manual_seed(9)
        w = torch.randn(32, 32)
        original = w.clone()
        nodes, history = train_path_nodes(w, fixed_nodes('cpu'), seed=17, iterations=4)
        self.assertLess(history[-1], history[0])
        self.assertTrue(all(b <= a + 1e-6 for a, b in zip(history, history[1:])))
        torch.testing.assert_close(w, original, atol=0, rtol=0)

    def test_outlier_base_and_importance_fit_preserve_inputs(self):
        torch.manual_seed(37)
        w = torch.randn(32, 32)
        h = torch.diag(torch.linspace(.1, 4, 32))
        original = w.clone()
        nodes = learn_outlier_nodes(w, h, seed=41, reserve=32,
                                    iterations=2, sample_pairs=512)
        self.assertEqual(tuple(nodes.shape), (256, 2))
        fitted, history = train_importance_nodes(w, h, nodes, seed=43, iterations=3)
        self.assertLessEqual(history[-1], history[0] + 1e-6)
        self.assertEqual(tuple(fitted.shape), (256, 2))
        torch.testing.assert_close(w, original, atol=0, rtol=0)

    def test_residual_specialists_never_increase_training_objective(self):
        torch.manual_seed(47)
        w = torch.randn(64, 32)
        h = torch.diag(torch.linspace(.2, 3, 32))
        original = w.clone()
        nodes, history = train_residual_specialists(
            w, h, fixed_nodes('cpu'), seed=53, reserve=16,
            iterations=2, sample_blocks=128)
        self.assertEqual(tuple(nodes.shape), (256, 2))
        self.assertLessEqual(history[-1]['loss'], history[0]['loss'])
        torch.testing.assert_close(w, original, atol=0, rtol=0)

    def test_neutral_implicit_path_remains_zero_after_training(self):
        torch.manual_seed(11)
        nodes, history = train_path_nodes(torch.randn(32, 32), fixed_nodes('cpu'),
                                         seed=19, iterations=3, neutral_zero=True)
        torch.testing.assert_close(decode_paths(torch.tensor([0]), nodes), torch.zeros(1, 8))
        self.assertLess(history[-1], history[0])

    def test_viterbi_matches_exhaustive_paths(self):
        torch.manual_seed(13)
        nodes = fixed_nodes('cpu')
        ids = torch.arange(65536)
        candidates = decode_paths(ids, nodes)
        x = torch.randn(3, 8)
        diagonal = torch.arange(1, 9).float()
        q, paths = viterbi(x, nodes, diagonal)
        costs = ((x[:, None] - candidates[None]).square() * diagonal).sum(-1)
        optimum = costs.min(1).values
        torch.testing.assert_close(((x - q).square() * diagonal).sum(1), optimum)
        torch.testing.assert_close(q, decode_paths(paths, nodes))

    def test_serialization_is_exactly_two_bits_including_table(self):
        ids = torch.randint(65536, (10, 100))
        ids.reshape(-1)[-FORCED_BLOCKS:] = 0
        nodes = fixed_nodes('cpu')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'layer.bin'
            save_layer(path, ids, nodes, 800, 17, .125)
            restored, decoded_nodes, seed, scale = load_layer(path)
            torch.testing.assert_close(restored, ids)
            torch.testing.assert_close(decoded_nodes, nodes)
            self.assertEqual((seed, scale), (17, .125))
            self.assertEqual(path.stat().st_size * 8 / (10 * 800), 2)


if __name__ == '__main__':
    unittest.main()
