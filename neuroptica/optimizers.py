from typing import List

import numpy as np

from neuroptica import MZILayer, OpticalMesh, PhaseShifterLayer


class Optimizer:

	def __init__(self, mesh: OpticalMesh):
		self.mesh = mesh

	def fit(self):
		pass

	def compute_phase_shifter_fields(self, X: np.ndarray) -> List[List[np.ndarray]]:
		'''
		Compute the foward-pass field at the end of each phase shifter in the mesh
		:param X: input field to the mesh
		:return: a list of (list of field values AFTER each phase shifter in a layer) for each layer
		'''

		fields = []

		X_current = np.copy(X)

		for layer in self.mesh.layers:

			if isinstance(layer, MZILayer):
				partial_transfer_matrices = layer.get_partial_transfer_matrices()
				theta_T = partial_transfer_matrices[1]
				phi_T = partial_transfer_matrices[3]
				fields.append([np.dot(theta_T, X_current), np.dot(phi_T, X_current)])

			elif isinstance(layer, PhaseShifterLayer):
				fields.append([np.dot(layer.get_transfer_matrix(), X_current)])

			X_current = np.dot(layer.get_transfer_matrix(), X_current)

		return fields

	def compute_adjoint_phase_shifter_fields(self, delta: np.ndarray) -> List[List[np.ndarray]]:
		'''
		Compute
		:param delta: input adjoint field to the mesh
		:return: a list of (list of field values BEFORE each phase shifter in a layer) for each layer. (Before = after
		on foward-pass) The ordering of the list is the opposite as in compute_phase_shifter_fields()
		'''

		adjoint_fields = []

		delta_current = np.copy(delta)

		for layer in reversed(self.mesh.layers):

			if isinstance(layer, MZILayer):
				partial_transfer_matrices = layer.get_partial_transfer_matrices()
				phi_T_inv = partial_transfer_matrices[1]
				theta_T_inv = partial_transfer_matrices[3]
				adjoint_fields.append([np.dot(phi_T_inv, delta_current), np.dot(theta_T_inv, delta_current)])

			elif isinstance(layer, PhaseShifterLayer):
				adjoint_fields.append([np.dot(layer.get_transfer_matrix(), delta_current)])

			delta_current = np.dot(layer.get_transfer_matrix().conj().T, delta_current)

		return adjoint_fields
