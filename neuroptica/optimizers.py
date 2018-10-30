from typing import Callable, List

import numpy as np

from neuroptica.layers import OpticalMeshNetworkLayer
from neuroptica.models import Sequential
from neuroptica.utils import pbar


class Optimizer:
    '''
    Base class for an optimizer
    '''

    def __init__(self, model: Sequential,
                 loss_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 d_loss_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        self.model = model
        self.loss_fn = loss_fn
        self.d_loss_fn = d_loss_fn

    def fit(self, data: List[np.ndarray], labels: List[np.ndarray], iterations=None, epochs=None, batch_size=None):
        raise NotImplementedError("must extend Optimizer.fit() method in child classes!")


class InSituGradientDescent(Optimizer):
    '''
    On-chip training with in-situ backpropagation using adjoint field method and standard gradient descent
    '''

    # def __init__(self, model: Sequential, loss_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    #              d_loss_fn: Callable[[np.ndarray, np.ndarray], np.ndarray], learning_rate=0.01):
    #     self.model = model
    #     self.loss_fn = loss_fn
    #     self.d_loss_fn = d_loss_fn
    #     self.learning_rate = learning_rate

    def fit(self, data: np.ndarray, labels: np.ndarray, iterations=1000, learning_rate=0.01,
            batch_size=32, show_progress=True):
        '''
        Fit the model to the labeled data
        :param data: features vector, shape: (n_features, n_samples)
        :param labels: labels vector, shape: (n_label_dim, n_samples)
        :param iterations:
        :param learning_rate:
        :param batch_size:
        :param show_progress:
        :return:
        '''
        
        losses = []

        n_features, n_samples = data.shape

        iterator = range(iterations)
        if show_progress: iterator = pbar(iterator)

        for iteration in iterator:

            iteration_losses = []

            for i in range(0, n_samples, batch_size):
                X = data[:, i:i + batch_size]
                Y = labels[:, i:i + batch_size]

                # Propagate the data forward
                Y_hat = self.model.forward_pass(X)
                d_loss = self.d_loss_fn(Y_hat, Y)
                iteration_losses.append(self.loss_fn(Y_hat, Y))

                # Compute the backpropagated signals for the model
                gradients = self.model.backward_pass(d_loss)
                delta_prev = d_loss  # backprop signal to send in the final layer

                # Compute the foward and adjoint fields at each phase shifter in all tunable layers
                for layer in reversed(self.model.layers):

                    if isinstance(layer, OpticalMeshNetworkLayer):
                        # Optimize the mesh using gradient descent
                        forward_field = np.mean(layer.X_prev, axis=1)
                        adjoint_field = np.mean(delta_prev, axis=1)
                        layer.mesh.adjoint_optimize(forward_field, adjoint_field, learning_rate)

                    # Set the backprop signal for the subsequent (spatially previous) layer
                    delta_prev = gradients[layer.__name__]

            losses.append(np.sum(iteration_losses) / n_samples)

        return losses


class InSituAdam:
    '''
    On-chip training with in-situ backpropagation using adjoint field method and adam optimizer
    '''

    def __init__(self,
                 model: Sequential,
                 loss_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 d_loss_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 step_size=0.01,
                 beta1=0.9,
                 beta2=0.99,
                 epsilon=1e-8):
        self.model = model
        self.loss_fn = loss_fn
        self.d_loss_fn = d_loss_fn
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def fit(self, data: List[np.ndarray], labels: List[np.ndarray], iterations=1000, epochs=None, batch_size=None,
            show_progress=True):
        losses = []
        iterator = range(iterations)
        if show_progress: iterator = pbar(iterator)

        for iteration in iterator:

            iteration_losses = []
            for X, Y in zip(data, labels):

                # Propagate the data forward
                Y_hat = self.model.forward_pass(X)
                d_loss = self.d_loss_fn(Y_hat, Y)
                iteration_losses.append(self.loss_fn(Y_hat, Y))

                # Compute the backpropagated signals for the model
                gradients = self.model.backward_pass(d_loss)
                delta_prev = d_loss  # backprop signal to send in the final layer

                # Compute the foward and adjoint fields at each phase shifter in all tunable layers
                for layer in reversed(self.model.layers):

                    if isinstance(layer, OpticalMeshNetworkLayer):
                        # Optimize the mesh using gradient descent
                        layer.mesh.adjoint_optimize(layer.X_prev, delta_prev, self.learning_rate)

                    # Set the backprop signal for the subsequent (spatially previous) layer
                    delta_prev = gradients[layer.__name__]

            losses.append(np.sum(iteration_losses) / len(iteration_losses))

        return losses
