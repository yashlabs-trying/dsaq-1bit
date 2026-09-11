import unittest
import numpy as np
from experiment import encode, decode, evaluate


class CodecTests(unittest.TestCase):
    def test_full_sparse_recovers_error_with_fp16_tolerance(self):
        x = np.random.default_rng(7).normal(size=(16, 8)).astype(np.float32)
        np.testing.assert_allclose(decode(encode(x, 0, x.size)), x, rtol=5e-4, atol=1e-5)

    def test_structured_residual_and_payload(self):
        rng = np.random.default_rng(8)
        x = (rng.normal(size=(128, 2)) @ rng.normal(size=(2, 16))).astype(np.float32)
        arrays = encode(x, 2, 0)
        metrics = evaluate(x, arrays)
        self.assertLess(metrics['residual_relative_l2'], .002)
        self.assertEqual(metrics['payload_bytes'], 16*2 + 2*16*2 + 128*2*2 + 16)

    def test_zero_and_invalid(self):
        x = np.zeros((8, 4), dtype=np.float32)
        np.testing.assert_array_equal(decode(encode(x, 0, 0)), x)
        with self.assertRaises(ValueError):
            encode(x, 5, 0)


if __name__ == '__main__':
    unittest.main()
