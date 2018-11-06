import unittest

from neuroptica.layers import *
from neuroptica.losses import MeanSquaredError
from neuroptica.models import Sequential
from neuroptica.nonlinearities import Abs
from neuroptica.optimizers import Optimizer
from tests.base import NeuropticaTest
from tests.test_models import TestModels


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

    def test_other_layers(self):
        for N in [4, 5]:
            layers = [DropMask(N, keep_ports=[0, 1])]

            for layer in layers:
                print("Testing layer {}".format(layer))

                batch_size = 11
                n_samples = batch_size * 3

                X_all = (2 * np.random.rand(N * n_samples) - 1).reshape((N, n_samples))
                Y_all = np.abs(X_all)

                if isinstance(layer, DropMask):
                    Y_all = Y_all[layer.ports, :]
                    model = Sequential([ClementsLayer(N), layer, Activation(Abs(N - len(layer.ports)))])

                else:
                    model = Sequential([ClementsLayer(N), layer, Activation(Abs(N))])

                # Use mean squared cost function
                loss = MeanSquaredError

                for X, Y in Optimizer.make_batches(X_all, Y_all, batch_size):
                    # Propagate the data forward
                    Y_hat = model.forward_pass(X)
                    d_loss = loss.dL(Y_hat, Y)

                    # Compute the backpropagated signals for the model
                    gradients = model.backward_pass(d_loss)

                    TestModels.verify_model_gradients(model, X, Y, loss.L, gradients, epsilon=1e-6)


if __name__ == "__main__":
    unittest.main()
