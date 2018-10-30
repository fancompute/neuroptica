import unittest

import numpy as np

from neuroptica.layers import Activation, ClementsLayer
from neuroptica.models import Sequential
from neuroptica.nonlinearities import Abs, AbsSquared, Mask
from tests.base import NeuropticaTest


class TestModels(NeuropticaTest):
    '''Tests for models'''

    def test_Sequential(self):
        '''Tests the z->|z| nonlinearity'''
        for N in [4, 5]:
            model = Sequential([
                ClementsLayer(N),
                Activation(Abs(N)),
                ClementsLayer(N),
                Activation(Abs(N)),
                ClementsLayer(N),
                Mask(N, mask=np.random.rand(N)),
                Activation(AbsSquared(N))
            ])
            # Check that the model behaves as expected for classifying vectorized inputs
            num_samples = 6
            X_data = self.random_complex_vector(N * num_samples).reshape((N, num_samples))
            Y_looped = np.array([model.forward_pass(X) for X in X_data.T]).T
            Y_vectorized = model.forward_pass(X_data)
            self.assert_allclose(Y_looped, Y_vectorized)


if __name__ == "__main__":
    unittest.main()
