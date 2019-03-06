'''This module contains a collection of optimizers for training neuroptica models to fit labeled data. All optimizers
starting with "InSitu" use the on-chip interferometric gradient calculation routine described in Hughes, et al. (2018),
"Training of photonic neural networks through in situ backpropagation and gradient measurement".'''

from typing import Tuple, Type

import numpy as np

from neuroptica.components import MZI, PhaseShifter
from neuroptica.layers import OpticalMeshNetworkLayer
from neuroptica.losses import Loss
from neuroptica.models import Sequential
from neuroptica.utils import pbar


class Optimizer:
    '''
    Base class for an optimizer
    '''

    def __init__(self, model: Sequential, loss: Type[Loss]):
        self.model = model
        self.loss = loss

    @staticmethod
    def make_batches(data: np.ndarray, labels: np.ndarray, batch_size: int,
                     shuffle=True) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Prepare batches of a given size from data and labels
        :param data: features vector, shape: (n_features, n_samples)
        :param labels: labels vector, shape: (n_label_dim, n_samples)
        :param batch_size: size of the batch
        :param shuffle: if true, batches will be randomized
        :return: yields a tuple (data_batch, label_batch)
        '''

        n_features, n_samples = data.shape

        batch_indices = np.arange(0, n_samples, batch_size)

        if shuffle:
            permutation = np.random.permutation(n_samples)
            data = data[:, permutation]  # this doesn't overwrite data from outside function call
            labels = labels[:, permutation]

        for i in batch_indices:
            X = data[:, i:i + batch_size]
            Y = labels[:, i:i + batch_size]
            yield X, Y

    def fit(self, data: np.ndarray, labels: np.ndarray, epochs=None, batch_size=None):
        raise NotImplementedError("must extend Optimizer.fit() method in child classes!")


class InSituGradientDescent(Optimizer):
    '''
    On-chip training with in-situ backpropagation using adjoint field method and standard gradient descent
    '''

    def __init__(self, model: Sequential, loss: Type[Loss], learning_rate=0.01):
        super().__init__(model, loss)
        self.learning_rate = learning_rate

    def fit(self, data: np.ndarray, labels: np.ndarray, epochs=1000, batch_size=32, show_progress=True):
        '''
        Fit the model to the labeled data
        :param data: features vector, shape: (n_features, n_samples)
        :param labels: labels vector, shape: (n_label_dim, n_samples)
        :param epochs:
        :param learning_rate:
        :param batch_size:
        :param show_progress:
        :return:
        '''

        losses = []

        n_features, n_samples = data.shape

        iterator = range(epochs)
        if show_progress: iterator = pbar(iterator)

        for epoch in iterator:

            total_epoch_loss = 0.0

            for X, Y in self.make_batches(data, labels, batch_size):

                # Propagate the data forward
                Y_hat = self.model.forward_pass(X)
                d_loss = self.loss.dL(Y_hat, Y)
                total_epoch_loss += np.sum(self.loss.L(Y_hat, Y))

                # Compute the backpropagated signals for the model
                gradients = self.model.backward_pass(d_loss)
                delta_prev = d_loss  # backprop signal to send in the final layer

                # Compute the foward and adjoint fields at each phase shifter in all tunable layers
                for layer in reversed(self.model.layers):
                    if isinstance(layer, OpticalMeshNetworkLayer):
                        layer.mesh.adjoint_optimize(layer.input_prev, delta_prev,
                                                    lambda dx: -1 * self.learning_rate * dx)

                    # Set the backprop signal for the subsequent (spatially previous) layer
                    delta_prev = gradients[layer.__name__]

            total_epoch_loss /= n_samples
            losses.append(total_epoch_loss)

            if show_progress:
                iterator.set_description("ℒ = {:.2f}".format(total_epoch_loss), refresh=False)

        return losses


class InSituAdam(Optimizer):
    '''
    On-chip training with in-situ backpropagation using adjoint field method and adam optimizer
    '''

    def __init__(self, model: Sequential, loss: Type[Loss], step_size=0.01, beta1=0.9, beta2=0.99, epsilon=1e-8):
        super().__init__(model, loss)
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}
        self.v = {}
        self.g = {}
        for layer in model.layers:
            if isinstance(layer, OpticalMeshNetworkLayer):
                for component in layer.mesh.all_tunable_components():
                    self.m[component] = np.zeros(component.dof)
                    self.v[component] = np.zeros(component.dof)
                    self.g[component] = np.zeros(component.dof)

    def fit(self, data: np.ndarray, labels: np.ndarray, epochs=1000, batch_size=32, show_progress=True,
            cache_fields=False, use_partial_vectors=False):
        '''
        Fit the model to the labeled data
        :param data: features vector, shape: (n_features, n_samples)
        :param labels: labels vector, shape: (n_label_dim, n_samples)
        :param epochs:
        :param batch_size:
        :param show_progress:
        :param cache_fields: if set to True, will cache fields at the phase shifters on the forward and backward pass
        :param use_partial_vectors: if set to True, the MZI partial matrices will be stored as Nx2 vectors
        :return:
        '''

        losses = []

        n_features, n_samples = data.shape

        iterator = range(epochs)
        if show_progress: iterator = pbar(iterator)

        for epoch in iterator:

            total_epoch_loss = 0.0
            batch = 0

            for X, Y in self.make_batches(data, labels, batch_size):

                batch += 1
                self.t += 1

                # Propagate the data forward
                Y_hat = self.model.forward_pass(X, cache_fields=cache_fields, use_partial_vectors=use_partial_vectors)
                d_loss = self.loss.dL(Y_hat, Y)
                total_epoch_loss += np.sum(self.loss.L(Y_hat, Y))

                # Compute the backpropagated signals for the model
                deltas = self.model.backward_pass(d_loss, cache_fields=cache_fields,
                                                  use_partial_vectors=use_partial_vectors)
                delta_prev = d_loss  # backprop signal to send in the final layer

                # Compute the foward and adjoint fields at each phase shifter in all tunable layers
                for layer in reversed(self.model.layers):
                    if isinstance(layer, OpticalMeshNetworkLayer):
                        gradients = layer.mesh.compute_gradients(layer.input_prev, delta_prev,
                                                                 cache_fields=cache_fields,
                                                                 use_partial_vectors=use_partial_vectors)
                        for cmpt in gradients:
                            self.g[cmpt] = np.mean(gradients[cmpt], axis=-1)
                            self.m[cmpt] = self.beta1 * self.m[cmpt] + (1 - self.beta1) * self.g[cmpt]
                            self.v[cmpt] = self.beta2 * self.v[cmpt] + (1 - self.beta2) * self.g[cmpt] ** 2
                            mhat = self.m[cmpt] / (1 - self.beta1 ** self.t)
                            vhat = self.v[cmpt] / (1 - self.beta2 ** self.t)

                            grad = -1 * self.step_size * mhat / (np.sqrt(vhat) + self.epsilon)

                            # Adjust settings by gradient amount
                            if isinstance(cmpt, PhaseShifter):
                                cmpt.phi += grad[0]

                            elif isinstance(cmpt, MZI):
                                dtheta, dphi = grad
                                cmpt.theta += dtheta
                                cmpt.phi += dphi

                    # Set the backprop signal for the subsequent (spatially previous) layer
                    delta_prev = deltas[layer.__name__]

            total_epoch_loss /= n_samples
            losses.append(total_epoch_loss)

            if show_progress:
                iterator.set_description("ℒ = {:.2f}".format(total_epoch_loss), refresh=False)

        return losses
