import unittest
import torch
from trellis_quant import decode_paths, fixed_nodes, viterbi
from trellis_beam import correlated_beam


class CorrelatedBeamTests(unittest.TestCase):
    def test_diagonal_metric_matches_optimal_viterbi(self):
        torch.manual_seed(21)
        x = torch.randn(3, 8)
        nodes = fixed_nodes('cpu')
        h = torch.diag(torch.arange(1, 9).float())
        oracle, _ = viterbi(x, nodes, h.diag())
        q, ids = correlated_beam(x, nodes, h, beam=2, fallback=False)
        torch.testing.assert_close(((x-q).square()*h.diag()).sum(1),
                                   ((x-oracle).square()*h.diag()).sum(1))
        torch.testing.assert_close(q, decode_paths(ids, nodes))

    def test_correlated_search_decodes_and_fallback_never_worsens_cost(self):
        torch.manual_seed(31)
        x = torch.randn(4, 16)
        nodes = fixed_nodes('cpu')
        a = torch.randn(16, 16)
        h = a.t() @ a + torch.eye(16)
        old, _ = viterbi(x, nodes, h.diag(), full_metric=h)
        for beam in (2, 4, 8):
            q, ids = correlated_beam(x, nodes, h, beam=beam)
            torch.testing.assert_close(q, decode_paths(ids, nodes, width=16))
            old_error, error = x-old, x-q
            self.assertTrue(torch.all(((error@h)*error).sum(1) <= ((old_error@h)*old_error).sum(1) + 1e-4))


if __name__ == '__main__':
    unittest.main()
