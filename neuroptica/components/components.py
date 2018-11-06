from typing import List

import numpy as np
from numpy import pi

from neuroptica.settings import NP_COMPLEX


# @jitclass(OrderedDict({
#     "ports": int32[:],
#     "tunable": optional(bool_),
#     "id": optional(int32)
# }))
class OpticalComponent:
    '''
    Base class for an optical component
    '''

    def __init__(self, ports: List[int], tunable: bool = True, dof: int = None, id: int = None):
        self.ports = ports
        self.tunable = tunable
        self.id = id
        self.dof = dof

    def __repr__(self):
        '''This should be overridden in child classes'''
        return '<OpticalComponent ports={}>'.format(self.ports)

    def get_transfer_matrix(self) -> np.ndarray:
        raise NotImplementedError("get_transfer_matrix() must be extended for child classes!")


_B = 1 / np.sqrt(2) * np.array([[1 + 0j, 0 + 1j], [0 + 1j, 1 + 0j]], dtype=NP_COMPLEX, order="C")


# @jitclass(OrderedDict({
#     "m": int32,
#     "n": int32
# }))
class Beamsplitter(OpticalComponent):
    '''
    Simulation of a perfect 50:50 beamsplitter
    '''

    def __init__(self, m: int, n: int):
        super().__init__([m, n], tunable=False, dof=0)
        self.m = m
        self.n = n

    def __repr__(self):
        return '<Beamsplitter, ports = {}>'.format(self.ports)

    def get_transfer_matrix(self) -> np.ndarray:
        return np.copy(_B)


# @jitclass(OrderedDict({
#     "m": int32,
#     "phi": optional(NUMBA_FLOAT)
# }))
class PhaseShifter(OpticalComponent):
    '''
    Single-mode phase shifter
    '''

    def __init__(self, m: int, phi: float = None):
        super().__init__([m], dof=1)
        self.m = m
        if phi is None: phi = 2 * pi * np.random.rand()
        self.phi = phi

    def __repr__(self):
        return '<PhaseShifter, port = {}, phi = {}>'.format(self.m, self.phi)

    def get_transfer_matrix(self) -> np.ndarray:
        return np.array([[np.exp(1j * self.phi)]], dtype=NP_COMPLEX)


# @jitclass(OrderedDict({
#     "m": int32,
#     "n": int32,
#     "phase_uncert": NUMBA_FLOAT,
#     "theta": optional(NUMBA_FLOAT),
#     "phi": optional(NUMBA_FLOAT)
# }))
class MZI(OpticalComponent):
    '''
    Simulation of a programmable phase-shifting Mach-Zehnder interferometer
    '''

    def __init__(self, m: int, n: int, theta: float = None, phi: float = None):
        super().__init__([m, n], dof=2)
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

        theta_shifter_matrix = np.array([[np.exp(1j * theta), 0 + 0j], [0 + 0j, 1 + 0j]], dtype=NP_COMPLEX)
        phi_shifter_matrix = np.array([[np.exp(1j * phi), 0 + 0j], [0 + 0j, 1 + 0j]], dtype=NP_COMPLEX)

        component_transfer_matrices = [_B, theta_shifter_matrix, _B, phi_shifter_matrix]

        if backward:
            component_transfer_matrices = [U.T for U in component_transfer_matrices[::-1]]

        if cumulative:
            T = component_transfer_matrices[0]
            partial_transfer_matrices = [T]
            for transfer_matrix in component_transfer_matrices[1:]:
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
