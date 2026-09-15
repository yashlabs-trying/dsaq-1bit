import unittest
import torch
from trellis_node_fit import parameter_map, fit_output_nodes
from trellis_quant import decode_paths, fixed_nodes


class OutputNodeFitTests(unittest.TestCase):
    def test_solver_matches_independent_dense_least_squares_oracle(self):
        torch.manual_seed(31)
        nodes = fixed_nodes('cpu')
        nodes[0] = 0
        paths = torch.randint(65536, (4, 1))
        w = torch.randn(4, 8)
        a = torch.randn(8, 8)
        h = a.t() @ a + torch.eye(8)
        mapping = parameter_map(paths, 8)
        active = mapping.unique()
        active = active[active >= 2]
        design = (mapping[..., None] == active).double()
        residual = (w - nodes.reshape(-1)[mapping]).double()
        normal = torch.einsum('rci,cd,rdj->ij', design, h.double(), design)
        rhs = torch.einsum('rci,cd,rd->i', design, h.double(), residual)
        optimum = torch.linalg.solve(normal, rhs)
        oracle_error = residual - design @ optimum
        oracle_loss = float(((oracle_error @ h.double()) * oracle_error).sum())
        fitted, report = fit_output_nodes(w, h, paths, nodes, width=8, iterations=64)
        self.assertLessEqual(report['fixed_path_output_loss_after'], oracle_loss + .005)

    def test_parameter_map_matches_decoder_including_signed_uint64_paths(self):
        nodes = fixed_nodes('cpu')
        paths = torch.tensor([[0, -1, -(2**63)], [123, 456, 789]])
        mapping = parameter_map(paths, 32)
        torch.testing.assert_close(nodes.reshape(-1)[mapping],
                                   decode_paths(paths, nodes, width=32).reshape(2, 96))

    def test_output_fitting_reduces_full_correlated_error_without_changing_weights(self):
        torch.manual_seed(13)
        nodes = fixed_nodes('cpu')
        nodes[0] = 0
        paths = torch.randint(65536, (6, 4))
        paths[0, -1] = 0
        w = torch.randn(6, 32)
        original = w.clone()
        a = torch.randn(32, 32)
        h = a.t() @ a + torch.eye(32)
        fitted, report = fit_output_nodes(w, h, paths, nodes, width=8, iterations=32)
        self.assertLess(report['fixed_path_output_loss_after'], report['fixed_path_output_loss_before'])
        torch.testing.assert_close(fitted[0], torch.zeros(2))
        torch.testing.assert_close(w, original, atol=0, rtol=0)
        error = w - fitted.reshape(-1)[parameter_map(paths, 8)]
        x = a
        expected = (error @ x.t()).square().sum() + error.square().sum()
        torch.testing.assert_close(((error @ h) * error).sum(), expected)


if __name__ == '__main__':
    unittest.main()
