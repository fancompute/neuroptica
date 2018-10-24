from functools import reduce
from typing import List, Type

from numpy import pi

from neuroptica.draw import *
from neuroptica.settings import NP_COMPLEX


class MZI:
	'''
	Simulation of a programmable phase-shifting Mach-Zehnder interferometer
	'''

	def __init__(self, N: int, m: int, n: int, inverted: bool = False,
				 theta: float = 2 * pi * np.random.rand(), phi: float = 2 * pi * np.random.rand()):
		self.N = N  # number of waveguides
		self.m = m  # input waveguide A index (0-indexed)
		self.n = n  # input waveguide B index
		self.inverted = inverted  # whether the MZI does Tmn or Tmn^-1
		self.phase_uncert = 0.005  # experimental phase uncertainty from MIT paper
		self.theta = theta
		self.phi = phi

	def __repr__(self):
		return 'MZI <theta={}, phi={}>'.format(self.theta, self.phi)

	def get_transfer_matrix(self, add_uncertainties=False) -> np.ndarray:

		if add_uncertainties:
			phi = self.phi + np.random.normal(0, self.phase_uncert)
			theta = self.theta + np.random.normal(0, self.phase_uncert)
		else:
			phi, theta = self.phi, self.theta

		if self.inverted:
			return np.array([
				[np.exp(-1j * phi) * np.cos(theta), np.exp(-1j * phi) * np.sin(theta)],
				[-1 * np.sin(theta), np.cos(theta)]
			])
		else:
			return np.array([
				[np.exp(1j * phi) * np.cos(theta), -1 * np.sin(theta)],
				[np.exp(1j * phi) * np.sin(theta), np.cos(theta)]
			])


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

	def __init__(self, N: int, components):
		self.N = N
		self.components = components

	def get_transfer_matrix(self):
		raise NotImplementedError("get_transfer_matrix() must be extended for child classes!")


class MZILayer(ComponentLayer):
	'''
	Represents a physical column of MZI's attached to an ensemble of waveguides
	'''

	def __init__(self, N: int, mzis: List[MZI]):
		super().__init__(N, mzis)
		self.N = N
		self.mzis = mzis

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


class OpticalMesh:
	'''
	Represents a mesh consisting of several layers of optical components
	'''

	def __init__(self, N: int, layers: List[Type[ComponentLayer]]):
		self.N = N
		self.layers = layers

	@staticmethod
	def verify_inputs(N, layers: List[Type[ComponentLayer]]):
		assert all([N == layer.N for layer in layers]), "Dimension mismatch in layers!"

	def get_transfer_matrix(self) -> np.ndarray:
		return reduce(np.dot, [layer.get_transfer_matrix() for layer in self.layers])

	def get_partial_transfer_matrices(self) -> List[np.ndarray]:
		'''Return the cumulative transfer matrices following each layer in the mesh'''
		partial_transfer_matrices = []
		T = np.eye(self.N, dtype=NP_COMPLEX)
		for layer in self.layers:
			T = np.dot(T, layer.get_transfer_matrix())
			partial_transfer_matrices.append(T)
		return partial_transfer_matrices
