import itertools
import tempfile
import unittest
from pathlib import Path
import torch
from trellis_quant import decode_paths, viterbi, fixed_nodes, save_layer, load_layer, FORCED_BLOCKS


class TrellisTests(unittest.TestCase):
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
