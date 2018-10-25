from typing import List

import numpy as np

from neuroptica import MZILayer, OpticalMesh, PhaseShifterLayer


class Optimizer:

	def __init__(self, mesh: OpticalMesh):
		self.mesh = mesh

	def fit(self):
		pass

	def compute_phase_shifter_fields(self, X: np.ndarray, align="right") -> List[List[np.ndarray]]:
		'''
		Compute the foward-pass field at the end of each phase shifter in the mesh
		:param X: input field to the mesh
		:return: a list of (list of field values to the left/right of each phase shifter in a layer) for each layer
		'''

		fields = []

		X_current = np.copy(X)

		for layer in self.mesh.layers:

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
		Compute
		:param delta: input adjoint field to the mesh
		:return: a list of (list of field values to the left/right of each phase shifter in a layer) for each layer
		The ordering of the list is the opposite as in compute_phase_shifter_fields()
		'''

		adjoint_fields = []

		delta_current = np.copy(delta)

		for layer in reversed(self.mesh.layers):

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
