import unittest
import tempfile
from pathlib import Path
import torch
from lattice_experiment import mixed_right, save_codes, load_codes, refine_codes, decode_e8p, HEADER


class MixedTransformTests(unittest.TestCase):
    def test_e8p_sign_parity_decode(self):
        packed = sum(9 << (4 * bit) for bit in range(8))
        grid = torch.full((256,), packed, dtype=torch.long)
        decoded = decode_e8p(torch.tensor([0, 1, 2, 3]), grid)
        expected = torch.full((4, 8), .75)
        expected[1] = .25
        expected[2] = .25
        expected[2, [0, 2]] = -.75
        expected[3, [0, 2]] = -.25
        torch.testing.assert_close(decoded, expected)

    def test_discrete_refinement_does_not_increase_hessian_loss(self):
        torch.manual_seed(24)
        class Codebook:
            grid = torch.randn(12, 8)
            def quantize(self, x):
                ids = torch.cdist(x, self.grid).argmin(1)
                return self.grid[ids], ids
        cb = Codebook()
        w = torch.randn(4, 24)
        a = torch.randn(24, 24)
        h = a.t() @ a + .1 * torch.eye(24)
        ids = torch.randint(0, 12, (4, 3))
        ids[0, -1] = 0
        q = cb.grid[ids].reshape_as(w)
        error = w - q
        before = ((error @ h) * error).sum()
        q, ids = refine_codes(w, h, q, ids, cb, forced=1, iters=2)
        error = w - q
        after = ((error @ h) * error).sum()
        self.assertLessEqual(after, before + 1e-4)
        self.assertEqual(int(ids[0, -1]), 0)
        torch.testing.assert_close(q, cb.grid[ids].reshape_as(w))

    def test_serialized_codes_with_implicit_positions(self):
        ids = torch.arange(48).reshape(3, 16)
        ids[0, -4:] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'layer.bin'
            save_codes(path, ids, 128, 12345, .125, 4)
            restored, seed, scale = load_codes(path, 4)
            torch.testing.assert_close(restored, ids)
            self.assertEqual(seed, 12345)
            self.assertEqual(scale, .125)
            self.assertEqual(path.stat().st_size, HEADER.size + 2 * (ids.numel() - 4))

    def test_qwen_widths_round_trip(self):
        torch.manual_seed(17)
        for width in (128, 896, 4864):
            x = torch.randn(3, width)
            signs = torch.randint(0, 2, (width,)).float() * 2 - 1
            transformed = mixed_right(x, signs)
            recovered = mixed_right(transformed, signs, inverse=True)
            torch.testing.assert_close(recovered, x, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(transformed.square().sum(1),
                                       x.square().sum(1), rtol=2e-5, atol=2e-4)

    def test_linear_output_preserved_by_two_sided_transform(self):
        torch.manual_seed(21)
        w, x = torch.randn(28, 152), torch.randn(5, 152)
        su = torch.randint(0, 2, (152,)).float() * 2 - 1
        sv = torch.randint(0, 2, (28,)).float() * 2 - 1
        wr = mixed_right(mixed_right(w.t(), sv).t(), su)
        transformed_output = mixed_right(x, su) @ wr.t()
        output = mixed_right(transformed_output, sv, inverse=True)
        torch.testing.assert_close(output, x @ w.t(), rtol=3e-5, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
