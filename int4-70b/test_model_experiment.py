import unittest
import torch
from model_experiment import packed_quant, unpack, size


class PackedQuantTests(unittest.TestCase):
    def test_levels_survive_bitpacking(self):
        for bits in (2, 4):
            x = torch.arange(64).remainder(2**bits).float().reshape(1,64)
            data, decoded = packed_quant(x,bits)
            torch.testing.assert_close(decoded,x,rtol=0,atol=0)
            torch.testing.assert_close(unpack(data,bits,'cpu'),x,rtol=0,atol=0)
            self.assertEqual(size(data),64*bits//8+4)

    def test_constant_group(self):
        x=torch.full((2,64),.5)
        _,decoded=packed_quant(x,2)
        torch.testing.assert_close(decoded,x,rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
