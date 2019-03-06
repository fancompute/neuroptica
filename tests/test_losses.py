import unittest

from neuroptica.layers import Activation, ClementsLayer
from neuroptica.losses import CategoricalCrossEntropy, MeanSquaredError
from neuroptica.models import Sequential
from neuroptica.nonlinearities import *
from neuroptica.optimizers import Optimizer
from tests.base import NeuropticaTest
from tests.test_models import TestModels


class TestLosses(NeuropticaTest):
    '''Tests for model losses'''

    def test_loss_gradients(self):

        N = 7
        losses = [MeanSquaredError, CategoricalCrossEntropy]

        for loss in losses:

            print("Testing loss {}".format(loss))

            batch_size = 6
            n_samples = batch_size * 4

            # Generate random points and label them (one-hot) according to index of max element
            X_all = (2 * np.random.rand(N * n_samples) - 1).reshape((N, n_samples))  # random N-D points
            X_max = np.argmax(X_all, axis=0)
            Y_all = np.zeros((N, n_samples))
            Y_all[X_max, np.arange(n_samples)] = 1.0

            # Make a single-layer model
            model = Sequential([
                ClementsLayer(N),
                Activation(AbsSquared(N))
            ])

            for X, Y in Optimizer.make_batches(X_all, Y_all, batch_size):
                # Propagate the data forward
                Y_hat = model.forward_pass(X)
                d_loss = loss.dL(Y_hat, Y)

                # Compute the backpropagated signals for the model
                gradients = model.backward_pass(d_loss)

                TestModels.verify_model_gradients(model, X, Y, loss.L, gradients, epsilon=1e-6)


if __name__ == "__main__":
    unittest.main()
