import tempfile
import unittest
from pathlib import Path
import torch
from trellis_quant import decode_paths, viterbi, fixed_nodes, save_layer, load_layer, FORCED_BLOCKS, train_path_nodes


class TrellisTests(unittest.TestCase):
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
