from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np
import qtrader


class TestValidator(unittest.TestCase):
    """Test `qtrader.utils._validator` methods."""

    def test__valid_type(self):
        """Test `qtrader.utils.valid_type` method."""
        a = ['a', 3, 'b', False]
        with self.assertRaises(TypeError):
            qtrader.framework.VALID_TYPE = True
            qtrader.utils.valid_type(a, int)
        qtrader.framework.VALID_TYPE = False
        return self.assertIsNone(qtrader.utils.valid_type(a, float))

    def test__valid_shape(self):
        """Test `qtrader.utils.valid_shape` method."""
        N = 10
        a = np.arange(N).reshape(-1, 1)
        b = list(range(N))
        with self.assertRaises(ValueError):
            qtrader.framework.VALID_SHAPE = True
            qtrader.utils.valid_shape(a, (N,))
        with self.assertRaises(AttributeError):
            qtrader.utils.valid_shape(b, (N,))
        qtrader.framework.VALID_SHAPE = False
        return self.assertIsNone(qtrader.utils.valid_shape(a, (N,)))


if __name__ == '__main__':
    unittest.main()
