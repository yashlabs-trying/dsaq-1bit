import unittest
import torch
from gptq import gptq_quantize, decode, payload_bytes


class GPTQTests(unittest.TestCase):
    def _run(self, out, cols, bits, groupwise=False, groupsize=-1, seed=0):
        torch.manual_seed(seed)
        W = torch.randn(out, cols)
        X = torch.randn(512, cols)
        H = 2 * X.t() @ X
        Q, codes, scale, zero = gptq_quantize(
            W, H, bits, groupwise=groupwise, groupsize=groupsize, mse=False)
        return W, Q, codes, scale, zero

    def test_decode_matches_output_perchannel(self):
        _, Q, codes, scale, zero = self._run(64, 128, 4)
        torch.testing.assert_close(decode(codes, scale, zero), Q, rtol=1e-5, atol=1e-5)

    def test_decode_matches_output_groupwise(self):
        _, Q, codes, scale, zero = self._run(64, 128, 3, groupwise=True, groupsize=32)
        self.assertEqual(scale.dim(), 2)
        self.assertEqual(scale.shape, (64, 4))
        torch.testing.assert_close(decode(codes, scale, zero), Q, rtol=1e-5, atol=1e-5)

    def test_gptq_reduces_output_error_vs_rtn(self):
        torch.manual_seed(1)
        out, cols, bits = 64, 128, 4
        W = torch.randn(out, cols)
        L = torch.randn(cols, 16)
        X = torch.randn(512, 16) @ L.t() + 0.05 * torch.randn(512, cols)
        H = 2 * X.t() @ X
        Q, codes, _, _ = gptq_quantize(W, H, bits, mse=False)
        maxq = 2 ** bits - 1
        xmin = torch.minimum(W.min(1)[0], torch.zeros(out))
        xmax = torch.maximum(W.max(1)[0], torch.zeros(out))
        s = (xmax - xmin) / maxq
        z = torch.round(-xmin / s)
        code = torch.clamp(torch.round(W / s.unsqueeze(1)) + z.unsqueeze(1), 0, maxq)
        W_rtn = s.unsqueeze(1) * (code - z.unsqueeze(1))
        err_gptq = (X @ (W - Q).t()).square().sum()
        err_rtn = (X @ (W - W_rtn).t()).square().sum()
        self.assertTrue((codes != code.to(torch.uint8)).any())
        self.assertLess(err_gptq, err_rtn)

    def test_payload_bytes(self):
        codes = torch.zeros(64, 128, dtype=torch.uint8)
        scale = torch.zeros(64)
        zero = torch.zeros(64)
        self.assertEqual(payload_bytes(codes, scale, zero, 2), (64 * 128 * 2 + 64 * 16 * 2) / 8)


if __name__ == '__main__':
    unittest.main()
