import unittest

from joint_mlp_selection import validate_selection_chain


def row(before, after):
    return dict(validation_before=dict(nll=before), validation_after=dict(nll=after))


class SelectionChainTests(unittest.TestCase):
    def test_accepts_monotone_continuous_chain(self):
        validate_selection_chain([row(4.0, 3.5), row(3.5, 3.5), row(3.5, 3.0)])

    def test_rejects_discontinuity(self):
        with self.assertRaisesRegex(ValueError, 'discontinuous'):
            validate_selection_chain([row(4.0, 3.5), row(3.6, 3.4)])

    def test_rejects_regression(self):
        with self.assertRaisesRegex(ValueError, 'increased'):
            validate_selection_chain([row(4.0, 4.1)])


if __name__ == '__main__':
    unittest.main()
