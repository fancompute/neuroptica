import unittest
from typing import Callable, Dict

import numpy as np

from neuroptica.component_layers import MZILayer, PhaseShifterLayer
from neuroptica.layers import Activation, ClementsLayer, OpticalMeshNetworkLayer
from neuroptica.models import Sequential
from neuroptica.nonlinearities import Abs, AbsSquared, LinearMask
from tests.base import NeuropticaTest


class TestModels(NeuropticaTest):
    '''Tests for models'''

    @staticmethod
    def verify_model_gradients(model: Sequential, X: np.ndarray, Y: np.ndarray,
                               loss_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                               deltas: Dict[str, np.ndarray], epsilon=1e-6, decimal=4):

        # Set initial backprop signal to d_loss
        delta_prev = deltas["output"]

        # Compute the foward and adjoint fields at each phase shifter in all tunable layers
        for net_layer in reversed(model.layers):

            if isinstance(net_layer, OpticalMeshNetworkLayer):
                # Optimize the mesh using gradient descent
                gradients = net_layer.mesh.compute_gradients(net_layer.input_prev, delta_prev)

                # Manually jiggle some phase shifters and check that the loss gradients are correct
                for layer in net_layer.mesh.layers:

                    if isinstance(layer, PhaseShifterLayer):

                        for phase_shifter in layer.phase_shifters:
                            dL_dphi_phase_shifter = gradients[phase_shifter][0]
                            delta_phis = epsilon * dL_dphi_phase_shifter

                            for i, delta_phi in enumerate(delta_phis):
                                # dL/dphi obtained from gradient computation
                                dL_dphi = dL_dphi_phase_shifter[i]

                                # estimate dL/dphi numerically
                                L_phi = loss_fn(model.forward_pass(X), Y)[i]
                                phase_shifter.phi += delta_phi
                                L_phi_plus_dphi = loss_fn(model.forward_pass(X), Y)[i]
                                phase_shifter.phi -= delta_phi  # revert change

                                dL_dphi_num = (L_phi_plus_dphi - L_phi) / (delta_phi + 1e-15)

                                np.testing.assert_almost_equal(dL_dphi, dL_dphi_num, decimal=decimal)


                    elif isinstance(layer, MZILayer):

                        for mzi in layer.mzis:
                            dL_dtheta_mzi, dL_dphi_mzi = gradients[mzi]
                            delta_thetas = epsilon * dL_dtheta_mzi
                            delta_phis = epsilon * dL_dphi_mzi

                            for i, delta_phi in enumerate(delta_phis):
                                # dL/dphi obtained from gradient computation
                                dL_dphi = dL_dphi_mzi[i]

                                # estimate dL/dphi numerically
                                L_phi = loss_fn(model.forward_pass(X), Y)[i]
                                mzi.phi += delta_phi
                                L_phi_plus_dphi = loss_fn(model.forward_pass(X), Y)[i]
                                mzi.phi -= delta_phi  # revert change

                                dL_dphi_num = (L_phi_plus_dphi - L_phi) / (delta_phi + 1e-15)

                                np.testing.assert_almost_equal(dL_dphi, dL_dphi_num, decimal=decimal)

                            for i, delta_theta in enumerate(delta_thetas):
                                # dL/dphi obtained from gradient computation
                                dL_dtheta = dL_dtheta_mzi[i]

                                # estimate dL/dphi numerically
                                L_theta = loss_fn(model.forward_pass(X), Y)[i]
                                mzi.theta += delta_theta
                                L_theta_plus_dtheta = loss_fn(model.forward_pass(X), Y)[i]
                                mzi.theta -= delta_theta  # revert change

                                dL_dtheta_num = (L_theta_plus_dtheta - L_theta) / (delta_theta + 1e-12)

                                np.testing.assert_almost_equal(dL_dtheta, dL_dtheta_num, decimal=decimal)

                    else:
                        raise ValueError("Tunable component layer must be phase-shifting!")

            # Set the backprop signal for the subsequent (spatially previous) layer
            delta_prev = deltas[net_layer.__name__]

    @staticmethod
    def verify_model_gradients_old(model: Sequential, X: np.ndarray, Y: np.ndarray,
                                   loss_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                                   gradients: Dict[str, np.ndarray], epsilon=1e-6, decimal=4):

        # Set initial backprop signal to d_loss
        delta_prev = gradients["output"]

        # Compute the foward and adjoint fields at each phase shifter in all tunable layers
        for net_layer in reversed(model.layers):

            if isinstance(net_layer, OpticalMeshNetworkLayer):
                # Optimize the mesh using gradient descent
                gradient_dict = net_layer.mesh.adjoint_optimize(net_layer.input_prev, delta_prev,
                                                                lambda dx: -1 * epsilon * dx, dry_run=True)

                # Manually jiggle some phase shifters and check that the loss gradients are correct
                for layer in net_layer.mesh.layers:

                    if isinstance(layer, PhaseShifterLayer):

                        dL_dphi_all = gradient_dict[layer][0]

                        for phase_shifter in layer.phase_shifters:
                            dL_dphi_phase_shifter = dL_dphi_all[phase_shifter.m]
                            delta_phis = epsilon * dL_dphi_phase_shifter

                            for i, delta_phi in enumerate(delta_phis):
                                # dL/dphi obtained from gradient computation
                                dL_dphi = dL_dphi_phase_shifter[i]

                                # estimate dL/dphi numerically
                                L_phi = loss_fn(model.forward_pass(X), Y)[i]
                                phase_shifter.phi += delta_phi
                                L_phi_plus_dphi = loss_fn(model.forward_pass(X), Y)[i]
                                phase_shifter.phi -= delta_phi  # revert change

                                dL_dphi_num = (L_phi_plus_dphi - L_phi) / (delta_phi + 1e-12)

                                np.testing.assert_almost_equal(dL_dphi, dL_dphi_num, decimal=decimal)


                    elif isinstance(layer, MZILayer):

                        dL_dtheta_all, dL_dphi_all = gradient_dict[layer]

                        for mzi in layer.mzis:
                            dL_dtheta_mzi = dL_dtheta_all[mzi.m]
                            dL_dphi_mzi = dL_dphi_all[mzi.m]
                            delta_thetas = epsilon * dL_dtheta_mzi
                            delta_phis = epsilon * dL_dphi_mzi

                            for i, delta_phi in enumerate(delta_phis):
                                # dL/dphi obtained from gradient computation
                                dL_dphi = dL_dphi_mzi[i]

                                # estimate dL/dphi numerically
                                L_phi = loss_fn(model.forward_pass(X), Y)[i]
                                mzi.phi += delta_phi
                                L_phi_plus_dphi = loss_fn(model.forward_pass(X), Y)[i]
                                mzi.phi -= delta_phi  # revert change

                                dL_dphi_num = (L_phi_plus_dphi - L_phi) / (delta_phi + 1e-12)

                                np.testing.assert_almost_equal(dL_dphi, dL_dphi_num, decimal=decimal)

                            for i, delta_theta in enumerate(delta_thetas):
                                # dL/dphi obtained from gradient computation
                                dL_dtheta = dL_dtheta_mzi[i]

                                # estimate dL/dphi numerically
                                L_theta = loss_fn(model.forward_pass(X), Y)[i]
                                mzi.theta += delta_theta
                                L_theta_plus_dtheta = loss_fn(model.forward_pass(X), Y)[i]
                                mzi.theta -= delta_theta  # revert change

                                dL_dtheta_num = (L_theta_plus_dtheta - L_theta) / (delta_theta + 1e-12)

                                np.testing.assert_almost_equal(dL_dtheta, dL_dtheta_num, decimal=decimal)

                    else:
                        raise ValueError("Tunable component layer must be phase-shifting!")

            # Set the backprop signal for the subsequent (spatially previous) layer
            delta_prev = gradients[net_layer.__name__]

    def test_Sequential(self):
        '''Tests the z->|z| nonlinearity'''
        for N in [4, 5]:
            model = Sequential([
                ClementsLayer(N),
                Activation(Abs(N)),
                ClementsLayer(N),
                Activation(Abs(N)),
                ClementsLayer(N),
                LinearMask(N, mask=np.random.rand(N)),
                Activation(AbsSquared(N))
            ])
            # Check that the model behaves as expected for classifying vectorized inputs
            num_samples = 100
            X_data = self.random_complex_vector(N * num_samples).reshape((N, num_samples))
            Y_looped = np.array([model.forward_pass(X) for X in X_data.T]).T
            Y_vectorized = model.forward_pass(X_data)
            self.assert_allclose(Y_looped, Y_vectorized)


if __name__ == "__main__":
    unittest.main()
