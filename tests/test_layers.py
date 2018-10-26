import unittest

import numpy as np

from neuroptica.layers import ClementsLayer, ReckLayer
from tests.base import NeuropticaTest


class TestLayers(NeuropticaTest):
    '''Tests for Network layers'''

    def test_ClementsLayer(self):
        '''Tests the Clements layer'''
        for N in [8, 9]:
            c = ClementsLayer(N)
            X = np.random.rand(N)
            X_out = c.forward_pass(X)
            # Check that a unitary transformation was done
            self.assert_allclose(np.linalg.norm(X), np.linalg.norm(X_out))

    def test_Reck(self):
        '''Tests the Reck layer'''
        for N in [8, 9]:
            r = ReckLayer(N)
            X = np.random.rand(N)
            X_out = r.forward_pass(X)
            # Check that a unitary transformation was done
            self.assert_allclose(np.linalg.norm(X), np.linalg.norm(X_out))


if __name__ == "__main__":
    unittest.main()
