from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest


class TestSetup(unittest.TestCase):
    """Test Dependencies from `requirements.txt`."""

    def test__numpy(self):
        import numpy as np
        return self.assertIsNotNone(np)

    def test__scipy(self):
        import scipy
        return self.assertIsNotNone(scipy)

    def test__pandas(self):
        import pandas as pd
        return self.assertIsNotNone(pd)

    def test__mpl(self):
        import matplotlib as mpl
        mpl.use('Agg')
        return self.assertIsNotNone(mpl)

    def test__sns(self):
        import seaborn as sns
        return self.assertIsNotNone(sns)

    def test__sklearn(self):
        import sklearn
        return self.assertIsNotNone(sklearn)

    def test__pytorch(self):
        import torch
        return self.assertIsNotNone(torch)

    def test__torchvision(self):
        import torchvision
        return self.assertIsNotNone(torchvision)

    def test__pandas_datareader(self):
        import pandas_datareader
        return self.assertIsNotNone(pandas_datareader)

    def test__qtrader(self):
        import qtrader
        return self.assertIsNotNone(qtrader)


if __name__ == '__main__':
    unittest.main()
