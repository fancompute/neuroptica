from typing import Callable, List, Type

import numpy as np

from neuroptica.components import MZILayer, PhaseShifterLayer
from neuroptica.layers import OpticalMeshNetworkLayer
from neuroptica.models import BaseModel, Sequential
from neuroptica.utils import pbar


class Optimizer:
    '''
    Base class for an optimizer
    '''

    def __init__(self, model: Type[BaseModel],
                 loss_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 d_loss_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        self.model = model
        self.loss_fn = loss_fn
        self.d_loss_fn = d_loss_fn

    def fit(self, data: List[np.ndarray], labels: List[np.ndarray], iterations=None, epochs=None, batch_size=None):
        raise NotImplementedError("must extend Optimizer.fit() method in child classes!")


class AdjointOptimizer:  # (Optimizer):
    '''
    On-chip training with in-situ backpropagation using adjoint field method
    '''

    def __init__(self, model: Sequential, loss_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 d_loss_fn: Callable[[np.ndarray, np.ndarray], np.ndarray], learning_rate=0.01):
        self.model = model
        self.loss_fn = loss_fn
        self.d_loss_fn = d_loss_fn
        self.learning_rate = learning_rate

    def optimize_mesh(self, layer: OpticalMeshNetworkLayer, X: np.ndarray, Y: np.ndarray):

        forward_fields = layer.mesh.compute_phase_shifter_fields(X, align="right")
        adjoint_fields = layer.mesh.compute_adjoint_phase_shifter_fields(Y, align="right")

        for component_layer, layer_fields, layer_fields_adj in \
                zip(layer.mesh.layers, forward_fields, reversed(adjoint_fields)):

            if isinstance(component_layer, PhaseShifterLayer):
                A_phi, A_phi_adj = layer_fields[0], layer_fields_adj[0]
                dL_dphi = -1 * np.imag(A_phi * A_phi_adj)
                for phase_shifter in component_layer.phase_shifters:
                    phase_shifter.phi -= self.learning_rate * dL_dphi[phase_shifter.m]

            elif isinstance(component_layer, MZILayer):
                A_theta, A_phi = layer_fields
                A_theta_adj, A_phi_adj = reversed(layer_fields_adj)
                dL_dtheta = -1 * np.imag(A_theta * A_theta_adj)
                dL_dphi = -1 * np.imag(A_phi * A_phi_adj)
                for mzi in component_layer.mzis:
                    mzi.theta -= self.learning_rate * dL_dtheta[mzi.m]
                    mzi.phi -= self.learning_rate * dL_dphi[mzi.m]

            else:
                raise ValueError("Tunable component layer must be phase-shifting!")

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
                        self.optimize_mesh(layer, layer.X_prev, delta_prev)

                    # Set the backprop signal for the subsequent (spatially previous) layer
                    delta_prev = gradients[layer.__name__]

            losses.append(np.sum(iteration_losses) / len(iteration_losses))

        return losses
