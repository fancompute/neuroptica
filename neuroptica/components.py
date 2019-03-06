'''The components submodule contains functionality for simulating individual optical components, such as a single
phase shifter, a beamsplitter, or an MZI. Components are combined in a :class:`~neuroptica.components.ComponentLayer`,
which describes the arrangement of the components on-chip.
'''

from typing import List

import numpy as np
from numba import jit
from numpy import pi

from neuroptica.settings import NP_COMPLEX


class OpticalComponent:
    '''Base class for an on-chip optical component'''

    def __init__(self, ports: List[int], tunable: bool = True, dof: int = None, id: int = None):
        '''
        Initialize the component

        :param list ports: list of ports the component is connected to
        :param bool tunable: whether the component is tunable or static
        :param int dof: number of degrees of freedom the component has
        :param int id: optional identifier for the component
        '''
        self.ports = ports
        self.tunable = tunable
        self.id = id
        self.dof = dof

    def __repr__(self):
        '''String representation, can be overridden in child classes'''
        return '<OpticalComponent ports={}>'.format(self.ports)

    def get_transfer_matrix(self) -> np.ndarray:
        '''Logic for computing the transfer operator of the component'''
        raise NotImplementedError("get_transfer_matrix() must be extended for child classes!")


_B = 1 / np.sqrt(2) * np.array([[1 + 0j, 0 + 1j], [0 + 1j, 1 + 0j]], dtype=NP_COMPLEX, order="C")


class Beamsplitter(OpticalComponent):
    '''Simulation of a perfect 50:50 beamsplitter'''

    def __init__(self, m: int, n: int):
        '''
        :param m: first waveguide index
        :param n: second waveguide index
        '''
        super().__init__([m, n], tunable=False, dof=0)
        self.m = m
        self.n = n

    def __repr__(self):
        return '<Beamsplitter, ports = {}>'.format(self.ports)

    def get_transfer_matrix(self) -> np.ndarray:
        return np.copy(_B)


class PhaseShifter(OpticalComponent):
    '''Single-mode phase shifter'''

    def __init__(self, m: int, phi: float = None):
        '''
        :param m: waveguide index
        :param phi: optional phase shift value; assigned randomly between [0, 2pi) if unspecified
        '''
        super().__init__([m], dof=1)
        self.m = m
        if phi is None: phi = 2 * pi * np.random.rand()
        self.phi = phi

    def __repr__(self):
        return '<PhaseShifter, port = {}, phi = {}>'.format(self.m, self.phi)

    def get_transfer_matrix(self) -> np.ndarray:
        return np.array([[np.exp(1j * self.phi)]], dtype=NP_COMPLEX)


class MZI(OpticalComponent):
    '''Simulation of a programmable phase-shifting Mach-Zehnder interferometer'''

    def __init__(self, m: int, n: int, theta: float = None, phi: float = None, phase_uncert=0.0):
        '''
        :param m: first waveguide index
        :param n: second waveguide index
        :param theta: phase shift value for inner phase shifter; assigned randomly between [0, 2pi) if unspecified
        :param phi: phase shift value for outer phase shifter; assigned randomly between [0, 2pi) if unspecified
        :param phase_uncertainty: optional uncertainty to add to the phase shifters; effective phase is computed as
        self.(theta, phi) + np.random.normal(0, self.phase_uncert) if add_uncertainties is set to True during simulation
        '''
        super().__init__([m, n], dof=2)
        self.m = m  # input waveguide A index (0-indexed)
        self.n = n  # input waveguide B index
        self.phase_uncert = phase_uncert
        if theta is None: theta = 2 * pi * np.random.rand()
        if phi is None: phi = 2 * pi * np.random.rand()
        self.theta = theta
        self.phi = phi

    def __repr__(self):
        return '<MZI theta={}, phi={}>'.format(self.theta, self.phi)

    def get_transfer_matrix(self, add_uncertainties=False) -> np.ndarray:
        '''
        Compute the transfer matrix for the tunable MZI given the current values of theta, phi
        :param add_uncertainties: whether to include uncertainties in the transfer matrix computation
        :return: the transfer matrix
        '''
        if add_uncertainties:
            phi = self.phi + np.random.normal(0, self.phase_uncert)
            theta = self.theta + np.random.normal(0, self.phase_uncert)
        else:
            phi, theta = self.phi, self.theta

        return 0.5 * np.array([
            [np.exp(1j * phi) * (np.exp(1j * theta) - 1), 1j * np.exp(1j * phi) * (1 + np.exp(1j * theta))],
            [1j * (np.exp(1j * theta) + 1), 1 - np.exp(1j * theta)]
        ], dtype=NP_COMPLEX)

    def get_partial_transfer_matrices(self, backward=False, cumulative=True, add_uncertainties=False) -> np.ndarray:
        '''
        Compute the partial transfer matrices of each "column" of the MZI -- after first beamsplitter, after first
        phase shifter, after second beamsplitter, and after second phase shifter
        :param backward: if true, compute the reverse transfer matrices in backward order
        :param cumulative: if true, each partial transfer matrix represents the total transfer matrix up to that point
        in the device
        :param add_uncertainties: whether to include uncertainties in the partial transfer matrix computation
        :return: numpy array of partial transfer matrices
        '''
        if add_uncertainties:
            phi = self.phi + np.random.normal(0, self.phase_uncert)
            theta = self.theta + np.random.normal(0, self.phase_uncert)
        else:
            theta, phi = self.theta, self.phi

        # return _get_mzi_partial_transfer_matrices(theta, phi, backward=backward, cumulative=cumulative)

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

            return np.array(partial_transfer_matrices)
        else:
            return np.array(component_transfer_matrices)


@jit(nopython=True, nogil=True, parallel=True)
def _get_mzi_partial_transfer_matrices(theta, phi, backward=False, cumulative=True):
    '''
    Deprecated function for accelerating partial transfer matrix computation with numba
    :param theta:
    :param phi:
    :param backward:
    :param cumulative:
    :return:
    '''
    theta_shifter_matrix = np.array([[np.exp(1j * theta), 0 + 0j], [0 + 0j, 1 + 0j]], dtype=NP_COMPLEX)
    phi_shifter_matrix = np.array([[np.exp(1j * phi), 0 + 0j], [0 + 0j, 1 + 0j]], dtype=NP_COMPLEX)

    component_transfer_matrices = np.empty((4, 2, 2), NP_COMPLEX)

    if not backward:
        component_transfer_matrices[0] = _B
        component_transfer_matrices[1] = theta_shifter_matrix
        component_transfer_matrices[2] = _B
        component_transfer_matrices[3] = phi_shifter_matrix
    else:
        component_transfer_matrices[3] = _B.T
        component_transfer_matrices[2] = theta_shifter_matrix.T
        component_transfer_matrices[1] = _B.T
        component_transfer_matrices[0] = phi_shifter_matrix.T

    if cumulative:
        T = component_transfer_matrices[0]
        partial_transfer_matrices = np.empty((4, 2, 2), NP_COMPLEX)
        partial_transfer_matrices[0] = component_transfer_matrices[0]
        for i in [1, 2, 3]:
            T = np.dot(component_transfer_matrices[i], T)
            partial_transfer_matrices[i] = T
        return partial_transfer_matrices

    else:
        return component_transfer_matrices
