import unittest

import numpy as np

from neuroptica.settings import NP_COMPLEX


class NeuropticaTest(unittest.TestCase):
    '''Tests for MZI meshes'''

    @staticmethod
    def assert_unitary(U, rtol=1e-3, atol=1e-6):
        np.testing.assert_allclose(U @ U.conj().T, np.eye(U.shape[0], dtype=NP_COMPLEX), rtol=rtol, atol=atol)

    @staticmethod
    def assert_almost_identity(U, rtol=1e-3, atol=1e-6):
        np.testing.assert_allclose(U, np.eye(U.shape[0], dtype=NP_COMPLEX), rtol=rtol, atol=atol)

    @staticmethod
    def assert_allclose(x, y, rtol=1e-3, atol=1e-6):
        np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)

    @staticmethod
    def random_complex_vector(N):
        return np.array(np.exp(np.random.rand(N)) * np.random.rand(N), dtype=NP_COMPLEX)
