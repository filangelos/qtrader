from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest


class TestSetup(unittest.TestCase):
    """Test Dependencies from `requirements.txt`."""

    def test_numpy(self):
        import numpy as np
        return self.assertIsNotNone(np)

    def test_scipy(self):
        import scipy
        return self.assertIsNotNone(scipy)

    def test_pandas(self):
        import pandas as pd
        return self.assertIsNotNone(pd)

    def test_mpl(self):
        import matplotlib as mpl
        mpl.use('Agg')
        return self.assertIsNotNone(mpl)

    def test_sns(self):
        import seaborn as sns
        return self.assertIsNotNone(sns)

    def test_sklearn(self):
        import sklearn
        return self.assertIsNotNone(sklearn)

    def test_tensorflow(self):
        import tensorflow as tf
        return self.assertIsNotNone(tf)

    def test_pandas_datareader(self):
        import pandas_datareader
        return self.assertIsNotNone(pandas_datareader)


if __name__ == '__main__':
    unittest.main()
