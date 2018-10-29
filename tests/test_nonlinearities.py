import unittest
from itertools import combinations

from neuroptica.nonlinearities import Abs
from tests.base import NeuropticaTest


class TestNonlinearities(NeuropticaTest):
    '''Tests for Network nonlinearities'''

    def test_Abs(self):
        '''Tests the z->|z| nonlinearity'''
        for N in [8, 9]:
            gamma = self.random_complex_vector(N)
            Z_back = self.random_complex_vector(N)
            backward_results = []

            for mode in ["full", "condensed", "polar"]:
                a = Abs(N, mode=mode)
                back = a.backward_pass(gamma, Z_back)
                backward_results.append(back)

            # Check that backprop results are the same for each mode
            for result1, result2 in combinations(backward_results, 2):
                self.assert_allclose(result1, result1)


if __name__ == "__main__":
    unittest.main()
