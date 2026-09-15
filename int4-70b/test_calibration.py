import unittest
import torch
from aq_block_experiment import collect_H


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 8)
        self.first = torch.nn.Linear(8, 8)
        self.second = torch.nn.Linear(8, 8)
        self.last = torch.nn.Linear(8, 8)
        self.last_calls = 0

    def forward(self, ids, use_cache=False):
        x = self.second(self.first(self.embedding(ids)))
        self.last_calls += 1
        return self.last(x)


@unittest.skipUnless(torch.cuda.is_available(), 'GPU calibration test')
class CalibrationTests(unittest.TestCase):
    @torch.inference_mode()
    def test_early_stop_and_gpu_products_preserve_hessians(self):
        torch.manual_seed(12)
        model = ToyModel().cuda().eval()
        layers = [('first', model.first), ('second', model.second)]
        tokenizer = lambda text, return_tensors: {'input_ids': torch.arange(64)[None]}
        reference = collect_H(model, layers, tokenizer, '', 32, 8)
        self.assertEqual(model.last_calls, 4)
        model.last_calls = 0
        fast = collect_H(model, layers, tokenizer, '', 32, 8,
                         gpu_hessian=True, stop_after_last=True)
        self.assertEqual(model.last_calls, 0)
        for name in reference:
            torch.testing.assert_close(reference[name], fast[name],
                                       atol=2e-5, rtol=2e-5)


if __name__ == '__main__':
    unittest.main()
