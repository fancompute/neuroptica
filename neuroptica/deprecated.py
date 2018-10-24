from functools import reduce

import SchemDraw as schem
import matplotlib.pyplot as plt
import scipy
from numpy import pi
from scipy.interpolate import interp1d
from tqdm import tqdm_notebook as pbar

from neuroptica.clements import *
from neuroptica.draw import *

_volts = np.linspace(0, 10, 81)
_phaseShifts = -1 * 2 * pi * np.array([
	0., -0.000210522, -0.000842266, -0.00189574, -0.00337178,
	-0.00527155, -0.00759657, -0.0103487, -0.0135302, -0.0171435,
	-0.0211915, -0.0256776, -0.0306052, -0.0359785, -0.0418016,
	-0.0480793, -0.0548168, -0.0620194, -0.069693, -0.077844, -0.086479,
	-0.095605, -0.10523, -0.115361, -0.126007, -0.137178, -0.148881,
	-0.161128, -0.173928, -0.187292, -0.201232, -0.21576, -0.230888,
	-0.246629, -0.262997, -0.280006, -0.297672, -0.316009, -0.335035,
	-0.354766, -0.375221, -0.396416, -0.418373, -0.441112, -0.464652,
	-0.489017, -0.514229, -0.540312, -0.567291, -0.595193, -0.624043,
	-0.653871, -0.684706, -0.716579, -0.749521, -0.783567, -0.818752,
	-0.855111, -0.892683, -0.931508, -0.971628, -1.01309, -1.05593,
	-1.1002, -1.14596, -1.19325, -1.24212, -1.29265, -1.34488, -1.39889,
	-1.45473, -1.51248, -1.57222, -1.63402, -1.69797, -1.76415, -1.83267,
	-1.90361, -1.9771, -2.05325, -2.13219
])
phaseShift = interp1d(_volts, _phaseShifts, kind='cubic')
inversePhaseShift = interp1d(_phaseShifts, _volts, kind='cubic')


class MZI:
	'''
	Class simulating a programmable phase-shifting Mach-Zehnder interferometer
	'''

	def __init__(self, N, m, n, inverted=False):
		self.N = N  # number of waveguides
		self.m = m  # input waveguide A index (0-indexed)
		self.n = n  # input waveguide B index
		self.inverted = inverted  # whether the MZI does Tmn or Tmn^-1
		self.phase_uncert = 0 * 0.005  # experimental phase uncertainty from MIT paper
		self.set_params(2 * pi * np.random.rand(), 2 * pi * np.random.rand())

	def set_params(self, theta, phi):
		self.theta = theta  # internal phase shift
		self.phi = phi  # external phase shift
		self.V_theta = self.phase_to_voltage(self.theta)  # voltage for internal modulator
		self.V_phi = self.phase_to_voltage(self.phi)  # voltage for external modulator

	def unitary(self):
		phi = self.phi + np.random.normal(0, self.phase_uncert)
		theta = self.theta + np.random.normal(0, self.phase_uncert)
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

	def operator(self):
		'''Expands self.unitary() to apply to N-dimensional set of waveguides'''
		U = self.unitary()
		m, n = self.m, self.n
		T = np.eye(self.N, dtype=np.complex64)
		T[m][m] = U[0, 0]
		T[m][n] = U[0, 1]
		T[n][m] = U[1, 0]
		T[n][n] = U[1, 1]
		return T

	def voltage_to_phase(self, V):
		return phaseShift(V)

	def phase_to_voltage(self, phi):
		return inversePhaseShift(phi)


class MZIBlock:
	'''
	Implements an arbitrary NxN unitary transformation with Clements decomposition of MZI's
	'''

	def __init__(self, N):
		self.N = N
		self.Tinv_block = []
		self.T_block = []
		# Initialize to random unitary matrix
		U_initial, _, _ = scipy.linalg.svd(np.random.rand(N, N))
		_, D, _, TinvCaches, Tcaches = clementsDecomposition(U_initial, show_progress=False)
		self.D = D
		for cache in TinvCaches:
			mzi = MZI(N, cache["m"], cache["n"], inverted=True)
			mzi.set_params(cache["theta"], cache["phi"])
			self.Tinv_block.append(mzi)
		for cache in Tcaches:
			mzi = MZI(N, cache["m"], cache["n"], inverted=False)
			mzi.set_params(cache["theta"], cache["phi"])
			self.T_block.append(mzi)

	def get_matrix(self):
		Tinv_list = [mzi.operator() for mzi in self.Tinv_block]
		T_list = [mzi.operator() for mzi in self.T_block]
		return reduce(np.dot, [*Tinv_list, self.D, *T_list])

	def set_matrix(self, U):
		'''Sets a new unitary with a Clements decomposition'''
		assert U.shape[0] == self.N and U.shape[1] == self.N, "Invalid matrix dimensions!"
		_, D, _, TinvCaches, Tcaches = clementsDecomposition(U)
		self.D = D
		for i, cache in enumerate(TinvCaches):
			self.Tinv_block[i].set_params(cache["theta"], cache["phi"])
		for i, cache in enumerate(Tcaches):
			self.T_block[i].set_params(cache["theta"], cache["phi"])

	def make_drawing(self, d=None, outputs=None):
		if d is None:
			d = schem.Drawing(fontsize=12)
		if outputs is None:
			outputs = [[np.array([0.0, i * wg_height])] for i in range(self.N)]
		if self.N > len(outputs):
			xmax = max([output[-1][0] for output in outputs])
			for i in range(len(outputs), self.N):  # augment output dimensions as needed
				outputs.append([np.array([xmax, i * wg_height])])
		elif self.N < len(outputs):
			outputs = outputs[0:self.N]
		# Draw the inverse block
		for mzi in self.Tinv_block:
			x, y = outputs[mzi.m][-1]
			if x < outputs[mzi.n][-1][0]:
				d.add(e.LINE, d='right', xy=[x, y], l=wg_width, color='red')
				x = outputs[mzi.n][-1][0]
			if outputs[mzi.n][-1][0] < x:
				d.add(e.LINE, d='right', xy=outputs[mzi.n][-1], l=wg_width, color='red')
			mzi_drawing = addMZI(d, mzi.V_theta, mzi.V_phi, xy=[x, y], color='red', inverted=True)
			outputs[mzi.m].append(mzi_drawing.out1)
			outputs[mzi.n].append(mzi_drawing.out2)
		d.add(e.LINE, d='right', xy=outputs[-1][-1], l=wg_width, color='red')
		outputs[-1][-1][0] += wg_width
		# Draw the diagonal phase shifters
		for i, element in enumerate(self.D.diagonal()):
			phase = np.angle(element) % (2 * pi)
			voltage = inversePhaseShift(phase)
			phaseShifter = addPhaseShifter(d, voltage, xy=outputs[i][-1], color='purple')
			outputs[i].append(phaseShifter.out)
		# Draw the T block
		for mzi in [*self.T_block]:
			x, y = outputs[mzi.m][-1]
			if x < outputs[mzi.n][-1][0]:
				d.add(e.LINE, d='right', xy=[x, y], l=wg_width, color='blue')
				x = outputs[mzi.n][-1][0]
			if outputs[mzi.n][-1][0] < x:
				d.add(e.LINE, d='right', xy=outputs[mzi.n][-1], l=wg_width, color='blue')
			mzi_drawing = addMZI(d, mzi.V_theta, mzi.V_phi, xy=[x, y], color='blue')
			outputs[mzi.m].append(mzi_drawing.out1)
			outputs[mzi.n].append(mzi_drawing.out2)
		# Draw the remaining lines to the right
		xmax = max([outputs[i][-1][0] for i in range(self.N)]) + 2
		for i in range(self.N):
			d.add(e.LINE, d='right', xy=outputs[i][-1], tox=xmax, color='blue')
			outputs[i].append([xmax, outputs[i][-1][1]])
		return d, outputs


class OIU:
	'''
	Simulation of an Optical Interference Unit (OIU). This implements the matrix multiplication
	in the synapses between a layer of two neurons. If the OIU is to simulate a NxN matrix with
	singular value decomposition W = U D V*, then N(N-1)/2 MZI's are needed for each of U and V*,
	and N gain elements are needed for the diagonal.
	'''

	def __init__(self, inputSize, outputSize):
		self.Ublock = MZIBlock(outputSize)
		self.D = np.eye(outputSize, inputSize)
		self.Vblock = MZIBlock(inputSize)

	def get_matrix(self):
		return self.Ublock.get_matrix() @ self.D @ self.Vblock.get_matrix()

	def set_matrix(self, newMatrix):
		U, Dvals, V = scipy.linalg.svd(newMatrix)
		D = scipy.linalg.diagsvd(Dvals, U.shape[0], V.shape[0])
		self.Ublock.set_matrix(U)
		self.D = D
		self.Vblock.set_matrix(V)

	def make_drawing(self, d=None, outputs=None):
		# Draw U block
		if d is None:
			d = schem.Drawing(fontsize=12)
		if outputs is not None:
			outputs = [[output[-1]] for output in outputs]
		d, outputs = self.Vblock.make_drawing(d, outputs)
		drawBox(d, outputs, label="$V^*$")
		# Draw gain
		outputs = [[output[-1]] for output in outputs]
		for i, diag in enumerate(self.D.diagonal()):
			gain = addGain(d, diag, xy=outputs[i][-1], color='magenta')
			outputs[i].append(gain.out)
		drawBox(d, outputs, label="$\\Sigma$")
		# Draw V block
		outputs = [[output[-1]] for output in outputs]
		d, outputs = self.Ublock.make_drawing(d, outputs)
		drawBox(d, outputs, label="$U$")
		return d, outputs


class NetworkLayer:
	'''
	Includes an OIU and an activation function
	'''

	def __init__(self, inputSize, outputSize, activation="relu"):
		self.inputSize = inputSize
		self.outputSize = outputSize
		oiu = OIU(inputSize, outputSize)
		self.oiu = oiu
		#         self.W = None
		self.bias = None
		self.activationType = activation
		self.cache = {
			"A_prev": None,
			"W": None,
			"b": None,
			"Z": None
		}

	def initialize(self):
		self.oiu.set_matrix(np.random.randn(self.outputSize, self.inputSize) * 0.01)
		#         self.W = np.random.randn(self.outputSize, self.inputSize) * 0.01
		self.bias = np.zeros((self.outputSize, 1))  # * 0.01

	def get_W(self):
		return self.oiu.get_matrix()

	def set_W(self, newW):
		self.oiu.set_matrix(newW)

	def forward_pass(self, A_prev):
		W = self.oiu.get_matrix()
		#         W = self.W
		Z = W @ A_prev + self.bias
		self.cache["W"] = W
		self.cache["A_prev"] = A_prev
		self.cache["b"] = self.bias
		self.cache["Z"] = Z
		return self.activation(Z)

	def activation(self, x):
		if self.activationType == "relu":
			return self.relu(x)
		elif self.activationType == "tanh":
			return self.tanh(x)
		elif self.activationType == "sigmoid":
			return self.sigmoid(x)
		else:
			raise ValueError("Invalid activation type: {}".format(self.activationType))

	def relu(self, x):
		return x * (x > 0)

	def tanh(self, x):
		return np.tanh(x)

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def backward_pass(self, dA):
		dZ = self.activation_backward(dA)
		dA_prev, dW, db = self.linear_backward(dZ)
		return dA_prev, dW, db

	def linear_backward(self, dZ):  # , A_prev, W, b):
		A_prev = self.cache["A_prev"]
		W = self.cache["W"]
		# b = self.cache["b"]

		m = A_prev.shape[1]
		dW = 1 / m * np.dot(dZ, A_prev.T)
		db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
		dA_prev = np.dot(W.T, dZ)
		return dA_prev, dW, db

	def activation_backward(self, dA):
		if self.activationType == "relu":
			return self.relu_backward(dA)
		elif self.activationType == "tanh":
			return self.tanh_backward(dA)
		elif self.activationType == "sigmoid":
			return self.sigmoid_backward(dA)
		else:
			raise ValueError("Invalid activation type: {}".format(self.activationType))

	def relu_backward(self, dA):
		dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
		Z = self.cache["Z"]
		# When z <= 0, you should set dz to 0 as well.
		dZ[Z <= 0] = 0
		return dZ

	def tanh_backward(self, dA):
		Z = self.cache["Z"]
		dZ = dA * (1 - np.tanh(Z) ** 2)
		return dZ

	def sigmoid_backward(self, dA):
		Z = self.cache["Z"]
		s = 1 / (1 + np.exp(-Z))
		dZ = dA * s * (1 - s)
		return dZ

	def make_drawing(self, d=None, outputs=None):
		# Draw U block
		if d is None:
			d = schem.Drawing(fontsize=12)
		d, outputs = self.oiu.make_drawing(d, outputs)
		# Draw activation
		outputs = [[output[-1]] for output in outputs]
		for i in range(self.oiu.get_matrix().shape[0]):
			activation = addActivation(d, self.activationType.upper(), xy=outputs[i][-1], color='green')
			outputs[i].append(activation.out)
		return d, outputs


class PhotonicNeuralNetwork:
	'''
	Simulation of a photonic neural network composed multiple layers, each of which includes
	an Optical Interference Unit (synapses) and Optical Nonlinearity Unit (activation)
	'''

	def __init__(self, layerSizes, activations):
		self.layers = []
		assert len(activations) == len(layerSizes) - 1, "Invalid activations dimensions!"
		#         if activations is None:
		#             activations = ["relu" for i in range(len(layerSizes)-1)]
		for i, activation in zip(range(len(layerSizes) - 1), activations):
			inputSize, outputSize = layerSizes[i], layerSizes[i + 1]
			self.layers.append(NetworkLayer(inputSize, outputSize, activation=activation))
		for layer in self.layers:
			layer.initialize()

	def forward_propagate(self, data):
		A_prev = data
		for layer in self.layers:
			A_prev = layer.forward_pass(A_prev)
		return A_prev

	def back_propagate(self, yhat, y):
		m = yhat.shape[1]
		y = y.reshape(yhat.shape)
		dA = - (np.divide(y, yhat) - np.divide(1 - y, 1 - yhat))
		# Set up gradient lists
		dW_list = [None for _ in self.layers]
		db_list = [None for _ in self.layers]
		for i in reversed(range(len(self.layers))):
			dA, dW, db = self.layers[i].backward_pass(dA)
			dW_list[i] = dW
			db_list[i] = db
		return dW_list, db_list

	def update_params(self, dW_list, db_list, learning_rate):
		for i in range(len(self.layers)):
			W = self.layers[i].cache["W"]
			b = self.layers[i].cache["b"]
			dW = dW_list[i]
			db = db_list[i]
			self.layers[i].set_W(W - learning_rate * dW)
			#             self.layers[i].W = (W - learning_rate * dW)
			self.layers[i].bias = b - learning_rate * db

	def compute_cost(self, yhat, y):
		'''Computes the cross-entropy loss between classification yhat and true labels y'''
		m = y.shape[1]  # number of examples
		cost = -1 / m * np.sum(np.multiply(np.log(yhat), y) + np.multiply(np.log(1 - yhat), 1 - y))
		return np.squeeze(cost)

	def learn(self, data, y, learning_rate, num_iterations, plot=True, showProgress=True):
		costs = []
		iterator = range(num_iterations)
		if showProgress:
			iterator = pbar(iterator)
		for iteration in iterator:
			yhat = self.forward_propagate(data)
			cost = self.compute_cost(yhat, y)
			costs.append(cost)
			dW_list, db_list = self.back_propagate(yhat, y)
			self.update_params(dW_list, db_list, learning_rate)
		if plot:
			plt.plot(costs)
			plt.ylabel('Cross-Entropy Loss')
			plt.xlabel('Iteration')
			plt.title("Learning rate = {}".format(learning_rate))
			plt.show()

	def classify(self, data):
		response = self.forward_propagate(data)
		return np.round(response)

	def make_drawing(self, d=None, outputs=None):
		# Draw U block
		if d is None:
			d = schem.Drawing(fontsize=12)
		for layer in self.layers:
			d, outputs = layer.make_drawing(d, outputs)
		return d, outputs
