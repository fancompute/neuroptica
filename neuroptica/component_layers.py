'''The componet_layers submodule contains functionality for assembling optical components on a simulated chip and
computing their transfer operations. A ComponentLayer represents a physical "column" of optical components which
acts on an input in parallel. ComponentLayers can be assembled into an OpticalMesh or put into a NetworkLayer.'''

from functools import reduce
from typing import Callable, Dict, Iterable, List, Type

import numpy as np
from numba import jit, prange

from neuroptica.components import MZI, OpticalComponent, PhaseShifter, _get_mzi_partial_transfer_matrices
from neuroptica.settings import NP_COMPLEX


class ComponentLayer:
    '''Base class for a single physical column of optical components which acts on inputs in parallel.'''

    def __init__(self, N: int, components: List[Type[OpticalComponent]]):
        '''
        Initialize the ComponentLayer
        :param int N: number of waveguides in the ComponentLayer
        :param list[OpticalComponent] components: list of
        '''
        self.N = N
        self.components = components

    def __iter__(self) -> Iterable[Type[OpticalComponent]]:
        '''Iterate over the optical components in the ComponentLayer'''
        yield from self.components

    def all_tunable_params(self) -> Iterable[float]:
        '''Enumerate all tunable parameters of the ComponentLayer; should be overwritten.'''
        raise NotImplementedError("all_tunable_params must be extended for child classes!")

    def get_transfer_matrix(self) -> np.ndarray:
        '''Enumerate all tunable parameters of the ComponentLayer; should be overwritten.'''
        raise NotImplementedError("get_transfer_matrix() must be extended for child classes!")


class MZILayer(ComponentLayer):
    '''Represents a physical column of MZI's attached to an ensemble of waveguides'''

    def __init__(self, N: int, mzis: List[MZI]):
        '''
        :param N: number of waveguides in the system the MZI layer is embedded
        :param mzis: list of MZIs in the column (can be less than N)
        '''
        super().__init__(N, mzis)
        self.mzis = mzis

    def __iter__(self) -> Iterable[MZI]:
        yield from self.mzis

    def all_tunable_params(self):
        for mzi in self.mzis:
            yield mzi.theta
            yield mzi.phi

    @classmethod
    def from_waveguide_indices(cls, N: int, waveguide_indices: List[int]):
        '''
        Create an MZI layer from a list of an even number of input/output indices. Each pair of waveguides in the
        iteration order will be assigned to an MZI
        :param N: size of MZILayer
        :param waveguide_indices: list of waveguides the layer attaches to
        :return: MZILayer class instance with size N and MZIs attached to waveguide_indices
        '''
        assert len(waveguide_indices) % 2 == 0 and len(waveguide_indices) <= N and \
               len(np.unique(waveguide_indices)) == len(waveguide_indices), \
            "Waveguide must have an even number <= N of indices which are all unique"
        mzis = []
        for i in range(0, len(waveguide_indices), 2):
            mzis.append(MZI(waveguide_indices[i], waveguide_indices[i + 1]))
        return cls(N, mzis)

    @staticmethod
    def verify_inputs(N: int, mzis: List[MZI]):
        '''
        Checks that the input MZIs are valid
        :param N: size of MZILayer
        :param mzis: list of MZIs
        '''
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

    def get_partial_transfer_matrices(self, backward=False, cumulative=True, add_uncertainties=False) -> np.ndarray:
        '''
        Return a list of 4 partial transfer matrices for the entire MZI layer corresponding to (1) after first BS in
        each MZI, (2) after theta shifter, (3) after second BS, and (4) after phi shifter. Order is reversed in the
        backwards case
        :param backward: whether to compute the backward partial transfer matrices
        :param cumulative: if true, each partial transfer matrix represents the total transfer matrix up to that point
        in the device
        :param add_uncertainties: whether to include uncertainties in transfer matrix computation
        :return: numpy array of partial transfer matrices
        '''

        Ttotal = np.eye(self.N, dtype=NP_COMPLEX)

        partial_transfer_matrices = []

        # Compute the (non-cumulative) partial transfer matrices for each MZI
        all_mzi_partials = [mzi.get_partial_transfer_matrices
                            (backward=backward, cumulative=False, add_uncertainties=add_uncertainties)
                            for mzi in self.mzis]

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

        return np.array(partial_transfer_matrices)

    def get_partial_transfer_vectors(self, backward=False, cumulative=True, add_uncertainties=False) -> np.ndarray:
        '''

        :param backward:
        :param cumulative:
        :param add_uncertainties:
        :return:
        '''
        Ttot = np.array([np.ones((self.N,), dtype=NP_COMPLEX), np.zeros((self.N,), dtype=NP_COMPLEX)])
        partial_transfer_vectors = []
        inds_mn = np.arange(self.N)

        # Compute the (non-cumulative) partial transfer matrices for each MZI
        all_mzi_partials = [mzi.get_partial_transfer_matrices
                            (backward=backward, cumulative=False, add_uncertainties=add_uncertainties)
                            for mzi in self.mzis]

        for depth in range(len(all_mzi_partials[0])):
            # Iterate over each sub-component at a given depth
            Tvec = np.array([np.ones((self.N,), dtype=NP_COMPLEX), np.zeros((self.N,), dtype=NP_COMPLEX)])

            for i, mzi in enumerate(self.mzis):
                U = all_mzi_partials[i][depth]
                m, n = mzi.m, mzi.n
                inds_mn[m] = n
                inds_mn[n] = m

                Tvec[0][m] = U[0, 0]
                Tvec[1][n] = U[0, 1]
                Tvec[1][m] = U[1, 0]
                Tvec[0][n] = U[1, 1]

            if cumulative:
                t1 = Tvec[0, :] * Ttot[0, :] + Tvec[1, :] * Ttot[1, inds_mn]
                t2 = Tvec[0, :] * Ttot[1, :] + Tvec[1, :] * Ttot[0, inds_mn]
                Ttot = np.vstack((t1, t2))
                partial_transfer_vectors.append(Ttot)
            else:
                partial_transfer_vectors.append(Tvec)

        return (partial_transfer_vectors, inds_mn)

    @staticmethod
    @jit(nopython=True, nogil=True, parallel=True)
    def _get_all_mzi_partials(thetaphis: List, backward=False, cumulative=True) -> np.ndarray:
        '''Deprecated method for accelerating partial transfer matrix computation with numba'''
        all_partials = np.empty((len(thetaphis), 4, 2, 2), NP_COMPLEX)
        for i in prange(len(thetaphis)):
            theta, phi = thetaphis[i]
            all_partials[i] = _get_mzi_partial_transfer_matrices(theta, phi, backward=backward, cumulative=cumulative)
        return all_partials

    @staticmethod
    @jit(nopython=True, nogil=True, parallel=True)
    def _get_partial_transfer_matrices(N, all_mzi_partials, mzi_mn, T_base, cumulative=True) -> np.ndarray:
        '''Deprecated method for accelerating partial transfer matrix computation with numba'''
        Ttotal = T_base.copy()

        partial_transfer_matrices = np.empty((len(all_mzi_partials[0]), N, N), NP_COMPLEX)

        for depth in range(len(all_mzi_partials[0])):
            # Iterate over each sub-component at a given depth
            T = T_base.copy()

            for i in range(len(mzi_mn)):
                U = all_mzi_partials[i][depth]
                m, n = mzi_mn[i]
                T[m][m] = U[0, 0]
                T[m][n] = U[0, 1]
                T[n][m] = U[1, 0]
                T[n][n] = U[1, 1]

            if cumulative:
                Ttotal = np.dot(T, Ttotal)
                partial_transfer_matrices[depth, :, :] = Ttotal
            else:
                partial_transfer_matrices[depth, :, :] = T

        return partial_transfer_matrices


class PhaseShifterLayer(ComponentLayer):
    '''Represents a column of N single-mode phase shifters'''

    def __init__(self, N: int, phase_shifters: List[PhaseShifter] = None):
        '''
        :param N: number of waveguides the column is embedded in
        :param phase_shifters: list of phase shifters in the column (can be less than N)
        '''
        super().__init__(N, phase_shifters)
        if phase_shifters is None:
            phase_shifters = [PhaseShifter(m) for m in range(N)]
        self.phase_shifters = phase_shifters

    def __iter__(self) -> Iterable[PhaseShifter]:
        yield from self.phase_shifters

    def all_tunable_params(self):
        for phase_shifter in self.phase_shifters:
            yield phase_shifter.phi

    def get_transfer_matrix(self, add_uncertainties=False) -> np.ndarray:
        T = np.eye(self.N, dtype=NP_COMPLEX)
        for phase_shifter in self.phase_shifters:
            m = phase_shifter.m
            T[m][m] = phase_shifter.get_transfer_matrix()[0, 0]
        return T


class OpticalMesh:
    '''Represents an optical "mesh" consisting of several layers of optical components, e.g. a rectangular MZI mesh'''

    def __init__(self, N: int, layers: List[Type[ComponentLayer]]):
        '''
        Initialize the OpticalMesh
        :param N: number of waveguides in the system the mesh is embedded in
        :param layers: list of ComponentLayers that the mesh contains (enumerates the columns of components)
        '''
        self.N = N
        self.layers = layers
        self.forward_fields = []
        self.adjoint_fields = []

    def __iter__(self) -> Iterable[ComponentLayer]:
        yield from self.layers

    def all_tunable_params(self) -> Iterable[float]:
        for layer in self.layers:
            for param in layer.all_tunable_params():
                yield param

    def all_tunable_components(self) -> Iterable[Type[OpticalComponent]]:
        for layer in self.layers:
            yield from layer

    @staticmethod
    def verify_inputs(N: int, layers: List[Type[ComponentLayer]]):
        assert all([N == layer.N for layer in layers]), "Dimension mismatch in layers!"

    def get_transfer_matrix(self) -> np.ndarray:
        return reduce(np.dot, [layer.get_transfer_matrix() for layer in reversed(self.layers)])

    def get_partial_transfer_matrices(self, backward=False, cumulative=True) -> List[np.ndarray]:
        '''
        Return the partial transfer matrices for the optical mesh after each column of components
        :param backward: whether to compute the backward partial transfer matrices
        :param cumulative: if true, each partial transfer matrix represents the total transfer matrix up to that point
        in the device
        :return: list of partial transfer matrices
        '''
        partial_transfer_matrices = []
        Ttotal = np.eye(self.N, dtype=NP_COMPLEX)
        layers = reversed(self.layers) if backward else self.layers
        for layer in layers:
            T = layer.get_transfer_matrix()
            if backward:
                T = T.T
            if cumulative:
                Ttotal = np.dot(T, Ttotal)  # needs to be (T . Ttotal), left multiply
                partial_transfer_matrices.append(Ttotal)
            else:
                partial_transfer_matrices.append(T)
        return partial_transfer_matrices

    def compute_phase_shifter_fields(self, X: np.ndarray, align="right", use_partial_vectors=False, include_bs=False) -> \
            List[List[np.ndarray]]:
        '''
        Compute the forward-propagating electric fields at the left/right of each phase shifter in the mesh
        :param X: input field to the mesh
        :param align: whether to align the fields at the beginning or end of each column
        :use_partial_vectors: can speed up the computation if set to True
        :include_bs: if true, also compute the phase shifter fields before/after each beamsplitter in the mesh
        :return: a list of (list of field values to the left/right of each phase shifter in a layer) for each layer
        '''

        fields = []

        X_current = np.copy(X)

        for layer in self.layers:

            if isinstance(layer, MZILayer):
                if use_partial_vectors:
                    if include_bs: raise NotImplementedError
                    (partial_transfer_vectors, inds_mn) = layer.get_partial_transfer_vectors(backward=False,
                                                                                             cumulative=True)
                    bs1_T, theta_T, bs2_T, phi_T = partial_transfer_vectors

                    if align == "right":
                        fields1 = theta_T[0, :][:, None] * X_current + theta_T[1, :][:, None] * X_current[inds_mn, :]
                        fields2 = phi_T[0, :][:, None] * X_current + phi_T[1, :][:, None] * X_current[inds_mn, :]
                        fields.append([fields1, fields2])
                    elif align == "left":
                        fields1 = bs1_T[0, :][:, None] * X_current + bs1_T[1, :][:, None] * X_current[inds_mn, :]
                        fields2 = bs2_T[0, :][:, None] * X_current + bs2_T[1, :][:, None] * X_current[inds_mn, :]
                        fields.append([fields1, fields2])
                    else:
                        raise ValueError('align must be "left" or "right"!')
                    X_current = phi_T[0, :][:, None] * X_current + phi_T[1, :][:, None] * X_current[inds_mn, :]
                else:
                    partial_transfer_matrices = layer.get_partial_transfer_matrices(backward=False, cumulative=True)
                    bs1_T, theta_T, bs2_T, phi_T = partial_transfer_matrices
                    if align == "right":
                        if include_bs:
                            fields.append([np.dot(bs1_T, X_current),
                                           np.dot(theta_T, X_current),
                                           np.dot(bs2_T, X_current),
                                           np.dot(phi_T, X_current)])
                        else:
                            if include_bs:
                                raise NotImplementedError
                            else:
                                fields.append([np.dot(theta_T, X_current), np.dot(phi_T, X_current)])
                    elif align == "left":
                        fields.append([np.dot(bs1_T, X_current), np.dot(bs2_T, X_current)])
                    else:
                        raise ValueError('align must be "left" or "right"!')
                    X_current = np.dot(phi_T, X_current)

            elif isinstance(layer, PhaseShifterLayer):
                if align == "right":
                    fields.append([np.dot(layer.get_transfer_matrix(), X_current)])
                elif align == "left":
                    fields.append([np.copy(X_current)])
                else:
                    raise ValueError('align must be "left" or "right"!')
                X_current = np.dot(layer.get_transfer_matrix(), X_current)

            else:
                raise TypeError("Layer is not instance of MZILayer or PhaseShifterLayer!")

        return fields

    def compute_adjoint_phase_shifter_fields(self, delta: np.ndarray, align="right",
                                             use_partial_vectors=False) -> List[List[np.ndarray]]:
        '''
        Compute the backward propagating (adjoint) electric fields at the left/right of each phase shifter in the mesh
        :param delta: input adjoint field to the mesh
        :param align: whether to align the fields at the beginning or end of each column
        :return: a list of (list of field values to the left/right of each phase shifter in a layer) for each layer
        The ordering of the list is the opposite as in compute_phase_shifter_fields()
        '''

        adjoint_fields = []

        delta_current = np.copy(delta)

        for layer in reversed(self.layers):

            if isinstance(layer, MZILayer):
                if use_partial_vectors:
                    (partial_transfer_vectors_inv, inds_mn) = layer.get_partial_transfer_vectors(backward=True,
                                                                                                 cumulative=True)
                    phi_T_inv, bs2_T_inv, theta_T_inv, bs1_T_inv = partial_transfer_vectors_inv

                    if align == "right":
                        fields2 = bs2_T_inv[0, :][:, None] * delta_current + bs2_T_inv[1, :][:, None] * delta_current[
                                                                                                        inds_mn, :]
                        adjoint_fields.append([np.copy(delta_current), fields2])
                    elif align == "left":
                        fields1 = phi_T_inv[0, :][:, None] * delta_current + phi_T_inv[1, :][:, None] * delta_current[
                                                                                                        inds_mn, :]
                        fields2 = theta_T_inv[0, :][:, None] * delta_current + theta_T_inv[1, :][:,
                                                                               None] * delta_current[inds_mn, :]
                        adjoint_fields.append([fields1, fields2])
                    else:
                        raise ValueError('align must be "left" or "right"!')
                    delta_current = bs1_T_inv[0, :][:, None] * delta_current + bs1_T_inv[1, :][:, None] * delta_current[
                                                                                                          inds_mn, :]

                else:
                    partial_transfer_matrices_inv = layer.get_partial_transfer_matrices(backward=True, cumulative=True)
                    phi_T_inv, bs2_T_inv, theta_T_inv, bs1_T_inv = partial_transfer_matrices_inv

                    if align == "right":
                        adjoint_fields.append([np.copy(delta_current), np.dot(bs2_T_inv, delta_current)])
                    elif align == "left":
                        adjoint_fields.append([np.dot(phi_T_inv, delta_current), np.dot(theta_T_inv, delta_current)])
                    else:
                        raise ValueError('align must be "left" or "right"!')
                    delta_current = np.dot(bs1_T_inv, delta_current)

            elif isinstance(layer, PhaseShifterLayer):
                if align == "right":
                    adjoint_fields.append([np.copy(delta_current)])
                elif align == "left":
                    adjoint_fields.append([np.dot(layer.get_transfer_matrix().T, delta_current)])
                else:
                    raise ValueError('align must be "left" or "right"!')
                # delta_current = np.dot(layer.get_transfer_matrix().T, delta_current)
                delta_current = np.dot(layer.get_transfer_matrix().T, delta_current)

            else:
                raise TypeError("Layer is not instance of MZILayer or PhaseShifterLayer!")

        return adjoint_fields

    def adjoint_optimize(self, forward_field: np.ndarray, adjoint_field: np.ndarray,
                         update_fn: Callable,  # update function takes a float and possibly other args and returns float
                         accumulator: Callable[[np.ndarray], float] = np.mean,
                         dry_run=False, cache_fields=False, use_partial_vectors=False):
        '''
        Compute the loss gradient as described in Hughes, et al (2018), "Training of photonic neural networks through
        in situ backpropagation and gradient measurement" and adjust the phase shifting parameters accordingly
        :param forward_field: forward-propagating input electric field at the beginning of the optical mesh
        :param adjoint_field: backward-propagating output electric field at the end of the optical mesh
        :param update_fn: a float=>float function to compute how to update parameters given a gradient
        :param accumulator: an array=>float function to compute how to compute a gradient from a batch of gradients;
        np.mean is used by default
        :param dry_run: if True, parameters will not be adjusted and a dictionary of parameter gradients for each
        ComponentLayer will be returned instead
        :param cache_fields: if True, forward and adjoint fields within the mesh will be cached
        :param use_partial_vectors: if True, uses partial vectors method to speed up transfer matrix computation
        :return: None, or (if dry_run==True) a dictionary of parameter gradients for each ComponentLayer
        '''

        if cache_fields:
            forward_fields = self.forward_fields
            adjoint_fields = self.adjoint_fields
        else:
            forward_fields = self.compute_phase_shifter_fields(forward_field, align="right",
                                                               use_partial_vectors=use_partial_vectors)
            adjoint_fields = self.compute_adjoint_phase_shifter_fields(adjoint_field, align="right",
                                                                       use_partial_vectors=use_partial_vectors)

        gradient_dict = {}

        for layer, layer_fields, layer_fields_adj in zip(self.layers, forward_fields, reversed(adjoint_fields)):

            if isinstance(layer, PhaseShifterLayer):
                A_phi, A_phi_adj = layer_fields[0], layer_fields_adj[0]
                dL_dphi = -1 * np.imag(A_phi * A_phi_adj)
                if dry_run:
                    gradient_dict[layer] = [dL_dphi]

                else:
                    for phase_shifter in layer.phase_shifters:
                        delta_phi = accumulator(dL_dphi[phase_shifter.m])
                        phase_shifter.phi += update_fn(delta_phi)

            elif isinstance(layer, MZILayer):
                A_theta, A_phi = layer_fields
                A_theta_adj, A_phi_adj = reversed(layer_fields_adj)
                dL_dtheta = -1 * np.imag(A_theta * A_theta_adj)
                dL_dphi = -1 * np.imag(A_phi * A_phi_adj)
                if dry_run:
                    gradient_dict[layer] = [dL_dtheta, dL_dphi]

                else:
                    for mzi in layer.mzis:
                        delta_theta = accumulator(dL_dtheta[mzi.m])
                        delta_phi = accumulator(dL_dphi[mzi.m])
                        mzi.theta += update_fn(delta_theta)
                        mzi.phi += update_fn(delta_phi)

            else:
                raise ValueError("Tunable component layer must be phase-shifting!")

        if dry_run:
            return gradient_dict

    def compute_gradients(self, forward_field: np.ndarray, adjoint_field: np.ndarray,
                          cache_fields=False, use_partial_vectors=False) \
            -> Dict[Type[OpticalComponent], np.ndarray]:
        '''
        Compute the gradients for each optical component within the mesh, without adjusting the parameters
        :param forward_field: forward-propagating input electric field at the beginning of the optical mesh
        :param adjoint_field: backward-propagating output electric field at the end of the optical mesh
        :param cache_fields: if True, forward and adjoint fields within the mesh will be cached
        :param use_partial_vectors: if True, uses partial vectors method to speed up transfer matrix computation
        :return:
        '''

        if cache_fields:
            forward_fields = self.forward_fields
            adjoint_fields = self.adjoint_fields
        else:
            forward_fields = self.compute_phase_shifter_fields(forward_field, align="right",
                                                               use_partial_vectors=use_partial_vectors)
            adjoint_fields = self.compute_adjoint_phase_shifter_fields(adjoint_field, align="right",
                                                                       use_partial_vectors=use_partial_vectors)

        gradients = {}

        for layer, layer_fields, layer_fields_adj in zip(self.layers, forward_fields, reversed(adjoint_fields)):

            if isinstance(layer, PhaseShifterLayer):
                A_phi, A_phi_adj = layer_fields[0], layer_fields_adj[0]
                dL_dphi = -1 * np.imag(A_phi * A_phi_adj)
                for phase_shifter in layer.phase_shifters:
                    gradients[phase_shifter] = np.array([dL_dphi[phase_shifter.m]])

            elif isinstance(layer, MZILayer):
                A_theta, A_phi = layer_fields
                A_theta_adj, A_phi_adj = reversed(layer_fields_adj)
                dL_dtheta = -1 * np.imag(A_theta * A_theta_adj)
                dL_dphi = -1 * np.imag(A_phi * A_phi_adj)
                for mzi in layer.mzis:
                    gradients[mzi] = np.array([dL_dtheta[mzi.m], dL_dphi[mzi.m]])

            else:
                raise ValueError("Tunable component layer must be phase-shifting!")

        return gradients
