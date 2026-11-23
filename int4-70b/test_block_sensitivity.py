import unittest

from block_sensitivity import layer_number


class BlockSensitivityHelpersTests(unittest.TestCase):
    def test_layer_number(self):
        self.assertEqual(layer_number('model.layers.0.self_attn.q_proj'), 0)
        self.assertEqual(layer_number('model.layers.23.mlp.down_proj'), 23)

    def test_layer_number_rejects_non_transformer_name(self):
        with self.assertRaises(ValueError):
            layer_number('lm_head')


if __name__ == '__main__':
    unittest.main()
