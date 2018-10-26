from functools import reduce
from typing import List, Type

import numpy as np
from numpy import pi

from neuroptica.settings import NP_COMPLEX


class OpticalComponent:
    '''
    Base class for an optical component
    '''

    def __init__(self, ports: List[int]):
        self.ports = ports

    def __repr__(self):
        '''This should be overridden in child classes'''
        return '<OpticalComponent ports={}>'.format(self.ports)

    def get_transfer_matrix(self) -> np.ndarray:
        raise NotImplementedError("get_transfer_matrix() must be extended for child classes!")


_B = 1 / np.sqrt(2) * np.array([[1, 1j], [1j, 1]], dtype=NP_COMPLEX)


class Beamsplitter(OpticalComponent):
    '''
    Simulation of a perfect 50:50 beamsplitter
    '''

    def __init__(self, m: int, n: int):
        super().__init__([m, n])
        self.m = m
        self.n = n

    def __repr__(self):
        return '<Beamsplitter, ports = {}>'.format(self.ports)

    def get_transfer_matrix(self) -> np.ndarray:
        return np.copy(_B)


class PhaseShifter(OpticalComponent):
    '''
    Single-mode phase shifter
    '''

    def __init__(self, m: int, phi: float = 2 * pi * np.random.rand()):
        super().__init__([m])
        self.m = m
        self.phi = phi

    def __repr__(self):
        return '<PhaseShifter, port = {}, phi = {}>'.format(self.m, self.phi)

    def get_transfer_matrix(self) -> np.ndarray:
        return np.array([[np.exp(1j * self.phi)]], dtype=NP_COMPLEX)


class MZI(OpticalComponent):
    '''
    Simulation of a programmable phase-shifting Mach-Zehnder interferometer
    '''

    def __init__(self, m: int, n: int, theta: float = None, phi: float = None):
        super().__init__([m, n])
        self.m = m  # input waveguide A index (0-indexed)
        self.n = n  # input waveguide B index
        # self.inverted = inverted  # whether the MZI does Tmn or Tmn^-1
        self.phase_uncert = 0.005  # experimental phase uncertainty from MIT paper
        if theta is None: theta = 2 * pi * np.random.rand()
        if phi is None: phi = 2 * pi * np.random.rand()
        self.theta = theta
        self.phi = phi

    def __repr__(self):
        return '<MZI theta={}, phi={}>'.format(self.theta, self.phi)

    def get_transfer_matrix(self, add_uncertainties=False) -> np.ndarray:

        if add_uncertainties:
            phi = self.phi + np.random.normal(0, self.phase_uncert)
            theta = self.theta + np.random.normal(0, self.phase_uncert)
        else:
            phi, theta = self.phi, self.theta

        return 0.5 * np.array([
            [np.exp(1j * phi) * (np.exp(1j * theta) - 1), 1j * np.exp(1j * phi) * (1 + np.exp(1j * theta))],
            [1j * (np.exp(1j * theta) + 1), 1 - np.exp(1j * theta)]
        ], dtype=NP_COMPLEX)

    def get_partial_transfer_matrices(self, backward=False, cumulative=True,
                                      add_uncertainties=False) -> List[np.ndarray]:

        if add_uncertainties:
            phi = self.phi + np.random.normal(0, self.phase_uncert)
            theta = self.theta + np.random.normal(0, self.phase_uncert)
        else:
            phi, theta = self.phi, self.theta

        partial_transfer_matrices = []
        theta_shifter_matrix = np.array([[np.exp(1j * theta), 0], [0, 1]], dtype=NP_COMPLEX)
        phi_shifter_matrix = np.array([[np.exp(1j * phi), 0], [0, 1]], dtype=NP_COMPLEX)

        component_transfer_matrices = [_B, theta_shifter_matrix, _B, phi_shifter_matrix]

        if backward:
            component_transfer_matrices = [U.conj().T for U in reversed(component_transfer_matrices)]

        if cumulative:
            T = np.eye(2, dtype=NP_COMPLEX)
            for transfer_matrix in component_transfer_matrices:
                T = np.dot(transfer_matrix, T)
                partial_transfer_matrices.append(T)

            return partial_transfer_matrices
        else:
            return component_transfer_matrices


# if self.inverted:
# 	return np.array([
# 		[np.exp(-1j * phi) * np.cos(theta), np.exp(-1j * phi) * np.sin(theta)],
# 		[-1 * np.sin(theta), np.cos(theta)]
# 	], dtype=NP_COMPLEX)
# else:
# 	return np.array([
# 		[np.exp(1j * phi) * np.cos(theta), -1 * np.sin(theta)],
# 		[np.exp(1j * phi) * np.sin(theta), np.cos(theta)]
# 	], dtype=NP_COMPLEX)
#
# def get_embedded_transfer_matrix(self, N: int) -> np.ndarray:
# 	'''Expands self.unitary() to apply to N-dimensional set of waveguides'''
# 	U = self.get_transfer_matrix()
# 	m, n = self.m, self.n
# 	T = np.eye(N, dtype=NP_COMPLEX)
# 	T[m][m] = U[0, 0]
# 	T[m][n] = U[0, 1]
# 	T[n][m] = U[1, 0]
# 	T[n][n] = U[1, 1]
# 	return T


class ComponentLayer:
    '''
    Base class for a physical column of optical components
    '''

    def __init__(self, N: int, components: List[Type[OpticalComponent]]):
        self.N = N
        self.components = components

    def get_transfer_matrix(self) -> np.ndarray:
        raise NotImplementedError("get_transfer_matrix() must be extended for child classes!")


class MZILayer(ComponentLayer):
    '''
    Represents a physical column of MZI's attached to an ensemble of waveguides
    '''

    def __init__(self, N: int, mzis: List[MZI]):
        super().__init__(N, mzis)
        self.mzis = mzis

    @classmethod
    def from_waveguide_indices(cls, N: int, waveguide_indices: List[int]):
        '''Create an MZI layer from a list of an even number of input/output indices. Each pair of waveguides in the
        iteration order will be assigned to an MZI'''
        assert len(waveguide_indices) % 2 == 0 and len(waveguide_indices) <= N and \
               len(np.unique(waveguide_indices)) == len(waveguide_indices), \
            "Waveguide must have an even number <= N of indices which are all unique"
        mzis = []
        for i in range(0, len(waveguide_indices), 2):
            mzis.append(MZI(waveguide_indices[i], waveguide_indices[i + 1]))
        return cls(N, mzis)

    @staticmethod
    def verify_inputs(N: int, mzis: List[MZI]):
        '''Checks that the input MZIs are valid'''
        assert len(mzis) <= N // 2, "Too many MZIs for layer with {} waveguides".format(N)
        input_ports = np.array([[mzi.m, mzi.n] for mzi in mzis]).flatten()
        assert len(np.unique(input_ports)) == len(input_ports), "MZIs share duplicate input ports!"

    def get_transfer_matrix(self, add_uncertainties=False) -> np.ndarray:
        T = np.eye(self.N, dtype=NP_COMPLEX)
        for mzi in self.mzis:
            U = mzi.get_transfer_matrix(add_uncertainties)
            m, n = mzi.m, mzi.n
            T[m][m] = U[0, 0]
            T[m][n] = U[0, 1]
            T[n][m] = U[1, 0]
            T[n][n] = U[1, 1]
        return T

    def get_partial_transfer_matrices(self, backward=False, cumulative=True,
                                      add_uncertainties=False) -> List[np.ndarray]:
        '''Return a list of 4 partial transfer matrices for the entire MZI layer corresponding to (1) after first BS in
        each MZI, (2) after theta shifter, (3) after second BS, and (4) after phi shifter. Order is reversed in the
        backwards case'''

        Ttotal = np.eye(self.N, dtype=NP_COMPLEX)

        partial_transfer_matrices = []
        # Compute the (non-cumulative) partial transfer matrices for each MZI
        all_mzi_partials = [mzi.get_partial_transfer_matrices(backward=backward, cumulative=False,
                                                              add_uncertainties=add_uncertainties) for mzi in self.mzis]

        for depth in range(len(all_mzi_partials[0])):
            # Iterate over each sub-component at a given depth
            T = np.eye(self.N, dtype=NP_COMPLEX)

            for i, mzi in enumerate(self.mzis):
                U = all_mzi_partials[i][depth]
                m, n = mzi.m, mzi.n
                T[m][m] = U[0, 0]
                T[m][n] = U[0, 1]
                T[n][m] = U[1, 0]
                T[n][n] = U[1, 1]

            if cumulative:
                Ttotal = np.dot(T, Ttotal)
                partial_transfer_matrices.append(Ttotal)
            else:
                partial_transfer_matrices.append(T)

        return partial_transfer_matrices


class PhaseShifterLayer(ComponentLayer):
    '''
    Represents a column of N single-mode phase shifters
    '''

    def __init__(self, N: int, phase_shifters: List[PhaseShifter] = None):
        super().__init__(N, phase_shifters)

        if phase_shifters is None:
            phase_shifters = [PhaseShifter(m) for m in range(N)]
        self.phase_shifters = phase_shifters

    def get_transfer_matrix(self, add_uncertainties=False) -> np.ndarray:
        T = np.eye(self.N, dtype=NP_COMPLEX)
        for phase_shifter in self.phase_shifters:
            m = phase_shifter.m
            T[m][m] = phase_shifter.get_transfer_matrix()[0, 0]
        return T


class OpticalMesh:
    '''
    Represents a mesh consisting of several layers of optical components
    '''

    def __init__(self, N: int, layers: List[Type[ComponentLayer]]):
        self.N = N
        self.layers = layers

    @staticmethod
    def verify_inputs(N: int, layers: List[Type[ComponentLayer]]):
        assert all([N == layer.N for layer in layers]), "Dimension mismatch in layers!"

    def get_transfer_matrix(self) -> np.ndarray:
        return reduce(np.dot, [layer.get_transfer_matrix() for layer in reversed(self.layers)])

    def get_partial_transfer_matrices(self, backward=False) -> List[np.ndarray]:
        '''Return the cumulative transfer matrices following each layer in the mesh'''
        partial_transfer_matrices = []
        Ttotal = np.eye(self.N, dtype=NP_COMPLEX)
        for layer in self.layers:
            T = layer.get_transfer_matrix()
            if backward:
                T = T.conj().T
            Ttotal = np.dot(T, Ttotal)  # needs to be (T . Ttotal), left multiply
            partial_transfer_matrices.append(Ttotal)
        return partial_transfer_matrices

    def compute_phase_shifter_fields(self, X: np.ndarray, align="right") -> List[List[np.ndarray]]:
        '''
        Compute the foward-pass field at the left/right of each phase shifter in the mesh
        :param X: input field to the mesh
        :return: a list of (list of field values to the left/right of each phase shifter in a layer) for each layer
        '''

        fields = []

        X_current = np.copy(X)

        for layer in self.layers:

            if isinstance(layer, MZILayer):
                partial_transfer_matrices = layer.get_partial_transfer_matrices(backward=False, cumulative=True)
                bs1_T, theta_T, bs2_T, phi_T = partial_transfer_matrices
                if align == "right":
                    fields.append([np.dot(theta_T, X_current), np.dot(phi_T, X_current)])
                elif align == "left":
                    fields.append([np.dot(bs1_T, X_current), np.dot(bs2_T, X_current)])
                else:
                    raise ValueError('align must be "left" or "right"!')

            elif isinstance(layer, PhaseShifterLayer):
                if align == "right":
                    fields.append([np.dot(layer.get_transfer_matrix(), X_current)])
                elif align == "left":
                    fields.append([np.copy(X_current)])
                else:
                    raise ValueError('align must be "left" or "right"!')

            else:
                raise TypeError("Layer is not instance of MZILayer or PhaseShifterLayer!")

            X_current = np.dot(layer.get_transfer_matrix(), X_current)

        return fields

    def compute_adjoint_phase_shifter_fields(self, delta: np.ndarray, align="right") -> List[List[np.ndarray]]:
        '''
        Compute the backward-pass field at the left/right of each phase shifter in the mesh
        :param delta: input adjoint field to the mesh
        :return: a list of (list of field values to the left/right of each phase shifter in a layer) for each layer
        The ordering of the list is the opposite as in compute_phase_shifter_fields()
        '''

        adjoint_fields = []

        delta_current = np.copy(delta)

        for layer in reversed(self.layers):

            if isinstance(layer, MZILayer):
                partial_transfer_matrices_inv = layer.get_partial_transfer_matrices(backward=True, cumulative=True)
                phi_T_inv, bs2_T_inv, theta_T_inv, bs1_T_inv = partial_transfer_matrices_inv

                if align == "right":
                    adjoint_fields.append([np.copy(delta_current), np.dot(bs2_T_inv, delta_current)])
                elif align == "left":
                    adjoint_fields.append([np.dot(phi_T_inv, delta_current), np.dot(theta_T_inv, delta_current)])
                else:
                    raise ValueError('align must be "left" or "right"!')

            elif isinstance(layer, PhaseShifterLayer):
                if align == "right":
                    adjoint_fields.append([np.copy(delta_current)])
                elif align == "left":
                    adjoint_fields.append([np.dot(layer.get_transfer_matrix().conj().T, delta_current)])
                else:
                    raise ValueError('align must be "left" or "right"!')

            else:
                raise TypeError("Layer is not instance of MZILayer or PhaseShifterLayer!")

            delta_current = np.dot(layer.get_transfer_matrix().conj().T, delta_current)

        return adjoint_fields
