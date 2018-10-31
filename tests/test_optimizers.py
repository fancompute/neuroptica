import unittest

from tests.base import NeuropticaTest


class TestOptimizers(NeuropticaTest):
    '''Tests for optimizers'''

    # def test_InSituGradientDescent(self):
    #     '''Tests the z->|z| nonlinearity'''
    #     for N in [4, 5]:
    #
    #         # Make a model
    #         model = Sequential([
    #             ClementsLayer(N),
    #             Activation(Abs(N)),
    #             ClementsLayer(N),
    #             Activation(Abs(N)),
    #             ClementsLayer(N),
    #             Mask(N, mask=np.random.rand(N)),
    #             Activation(AbsSquared(N))
    #         ])
    #
    #         # Prepare data
    #         X, Y = generate_diagonal_planar_dataset(seed=1)
    #         P0 = 10
    #         X_formatted = np.pad(X, (0, N - 2), mode="constant")
    #         for i, x in enumerate(X_formatted):
    #             X_formatted[i][2] = np.sqrt(P0 - np.sum(x ** 2))
    #         Y_formatted = np.pad(Y, (0, N - 2), mode="constant")
    #
    #         X_formatted = X_formatted.T
    #         Y_formatted = Y_formatted.T
    #
    #
    #         X_data = self.random_complex_vector(N * num_samples).reshape((N, num_samples))
    #         Y_looped = np.array([model.forward_pass(X) for X in X_data.T]).T
    #         Y_vectorized = model.forward_pass(X_data)
    #         self.assert_allclose(Y_looped, Y_vectorized)


if __name__ == "__main__":
    unittest.main()
