import unittest

import numpy as np

from neuroptica import NP_COMPLEX
from neuroptica.components import MZI, MZILayer, OpticalMesh, PhaseShifter, PhaseShifterLayer
from tests.base import NeuropticaTest


class TestComponents(NeuropticaTest):
    '''Tests for MZI meshes'''

    def test_MZI(self):
        '''Tests an invidual MZI'''
        m = MZI(0, 1)

        # Should be unitary
        self.assert_unitary(m.get_transfer_matrix())

        # Test partial transfer matrices
        partial_transfers_forward = m.get_partial_transfer_matrices()
        for T in partial_transfers_forward:
            self.assert_unitary(T)
        self.assert_allclose(partial_transfers_forward[-1], m.get_transfer_matrix())

        partial_transfers_backward = m.get_partial_transfer_matrices(backward=True)
        for T in partial_transfers_backward:
            self.assert_unitary(T)
        self.assert_allclose(partial_transfers_backward[-1], m.get_transfer_matrix().conj().T)

        # Test cross case
        m.theta = 0.0
        m.phi = 0.0
        self.assert_allclose(m.get_transfer_matrix(), np.array([[0, 1j], [1j, 0]]))

        # Test bar case
        m.theta = np.pi
        m.phi = np.pi
        self.assert_almost_identity(m.get_transfer_matrix())

    def test_PhaseShifter(self):
        '''Tests for an individual phase shifter'''
        p = PhaseShifter(0)

        for _ in range(5):
            phi = 2 * np.pi * np.random.rand()
            p.phi = phi
            self.assert_allclose(p.get_transfer_matrix(), np.array([[np.exp(1j * phi)]], dtype=NP_COMPLEX))

    def test_MZILayer(self):
        '''Test for the MZILayer class'''
        # Test bar case
        N = 4
        mzis = [MZI(i, i + 1, theta=np.pi, phi=np.pi) for i in range(0, N, 2)]
        l = MZILayer(N, mzis)
        self.assert_almost_identity(l.get_transfer_matrix())

        # Test odd case and from_waveguide_indices()
        N = 5
        l = MZILayer.from_waveguide_indices(N, list(range(1, N)))
        self.assert_unitary(l.get_transfer_matrix())

        partial_transfers_forward = l.get_partial_transfer_matrices()
        for T in partial_transfers_forward:
            self.assert_unitary(T)
        self.assert_allclose(partial_transfers_forward[-1], l.get_transfer_matrix())

        partial_transfers_backward = l.get_partial_transfer_matrices(backward=True)
        for T in partial_transfers_backward:
            self.assert_unitary(T)
        self.assert_allclose(partial_transfers_backward[-1], l.get_transfer_matrix().conj().T)

    def test_PhaseShifterLayer(self):
        '''Tests for the PhaseShifterLayer class'''
        N = 4
        phase_shifters = [PhaseShifter(m, phi=0) for m in range(N)]
        p = PhaseShifterLayer(N, phase_shifters)

        self.assert_allclose(p.get_transfer_matrix(), np.eye(N))

        N = 5
        p = PhaseShifterLayer(N, phase_shifters=None)

        self.assert_unitary(p.get_transfer_matrix())

    def test_OpticalMesh(self):
        '''Tests for the OpticalMesh class'''
        for N in [8, 9]:
            l1 = PhaseShifterLayer(N)
            l2 = MZILayer.from_waveguide_indices(N, list(range(N % 2, N)))
            m = OpticalMesh(N, [l1, l2])

            self.assert_unitary(m.get_transfer_matrix())

            for T in m.get_partial_transfer_matrices():
                self.assert_unitary(T)

            for T in m.get_partial_transfer_matrices(backward=True):
                self.assert_unitary(T)

            X_in = np.random.rand(N)
            X_out = np.dot(m.get_transfer_matrix(), X_in)

            fields = m.compute_phase_shifter_fields(X_in, align="right")
            adjoint_fields = m.compute_adjoint_phase_shifter_fields(X_out, align="right")

            # Check that a unitary transformation was done
            for layer_fields in fields:
                for component_fields in layer_fields:
                    self.assert_allclose(np.linalg.norm(X_in), np.linalg.norm(component_fields))

            # Check results match at end
            output_fields = fields[-1][-1]
            self.assert_allclose(X_out, output_fields)

            # Check that a unitary transformation was done
            for layer_fields_adj in adjoint_fields:
                for component_fields_adj in layer_fields_adj:
                    self.assert_allclose(np.linalg.norm(X_in), np.linalg.norm(component_fields_adj))

            # Check results match at end
            output_fields_adj = m.compute_adjoint_phase_shifter_fields(X_out, align="left")[-1][-1]
            self.assert_allclose(output_fields_adj, X_in)

            # Check that adjoint field of X_out equals regular field of X_in
            for layer_fields, layer_fields_adj in zip(fields, reversed(adjoint_fields)):
                for component_fields, component_fields_adj in zip(layer_fields, reversed(layer_fields_adj)):
                    self.assert_allclose(component_fields, component_fields_adj)


if __name__ == "__main__":
    unittest.main()
