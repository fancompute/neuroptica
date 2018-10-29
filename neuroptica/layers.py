import numpy as np

from neuroptica.components import MZILayer, OpticalMesh, PhaseShifterLayer
from neuroptica.nonlinearities import ComplexNonlinearity


class NetworkLayer:
    '''
    Represents a logical layer in a neural network (different from ComponentLayer)
    '''

    def __init__(self, input_size: int, output_size: int, initializer=None):
        self.input_size = input_size
        self.output_size = output_size
        self.initializer = initializer
        self.X_prev: np.ndarray = None
        self.Z_prev: np.ndarray = None

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError('forward_pass() must be overridden in child class!')

    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        raise NotImplementedError('backward_pass() must be overridden in child class!')


class Activation(NetworkLayer):

    def __init__(self, nonlinearity: ComplexNonlinearity):
        super().__init__(nonlinearity.N, nonlinearity.N)
        self.nonlinearity = nonlinearity

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        self.X_prev = X
        self.Z_prev = self.nonlinearity.forward_pass(X)
        return self.Z_prev

    def backward_pass(self, gamma: np.ndarray) -> np.ndarray:
        return self.nonlinearity.backward_pass(gamma, self.Z_prev)


class OpticalMeshNetworkLayer(NetworkLayer):
    '''
    Base class for any network layer consisting of an optical mesh of phase shifters and MZIs
    '''

    def __init__(self, input_size: int, output_size: int, initializer=None):
        super().__init__(input_size, output_size, initializer=initializer)
        self.mesh: OpticalMesh = None

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError('forward_pass() must be overridden in child class!')

    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        raise NotImplementedError('backward_pass() must be overridden in child class!')


class ClementsLayer(OpticalMeshNetworkLayer):
    '''
    Performs a unitary NxM operator with MZIs arranged in a Clements decomposition. If M=N then the layer can perform
    any arbitrary unitary operator
    '''

    def __init__(self, N: int, M=None, include_phase_shifter_layer=True, initializer=None):
        super().__init__(N, N, initializer=initializer)

        layers = []
        if include_phase_shifter_layer:
            layers.append(PhaseShifterLayer(N))

        if M is None:
            M = N

        for layer_index in range(M):
            if N % 2 == 0:  # even number of waveguides
                if layer_index % 2 == 0:
                    layers.append(MZILayer.from_waveguide_indices(N, list(range(0, N))))
                else:
                    layers.append(MZILayer.from_waveguide_indices(N, list(range(1, N - 1))))
            else:  # odd number of waveguides
                if layer_index % 2 == 0:
                    layers.append(MZILayer.from_waveguide_indices(N, list(range(0, N - 1))))
                else:
                    layers.append(MZILayer.from_waveguide_indices(N, list(range(1, N))))

        self.mesh = OpticalMesh(N, layers)

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        self.X_prev = X
        self.Z_prev = np.dot(self.mesh.get_transfer_matrix(), X)
        return self.Z_prev

    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        return np.dot(self.mesh.get_transfer_matrix().conj().T, delta)


class ReckLayer(OpticalMeshNetworkLayer):
    '''
    Performs a unitary NxN operator with MZIs arranged in a Reck decomposition
    '''

    def __init__(self, N: int, include_phase_shifter_layer=True, initializer=None):
        super().__init__(N, N, initializer=initializer)

        layers = []
        if include_phase_shifter_layer:
            layers.append(PhaseShifterLayer(N))

        mzi_limits_upper = [i for i in range(1, N)] + [i for i in range(N - 2, 1 - 1, -1)]
        mzi_limits_lower = [(i + 1) % 2 for i in mzi_limits_upper]

        print(mzi_limits_upper, mzi_limits_lower)

        for start, end in zip(mzi_limits_lower, mzi_limits_upper):
            layers.append(MZILayer.from_waveguide_indices(N, list(range(start, end + 1))))

        self.mesh = OpticalMesh(N, layers)

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        self.X_prev = X
        self.Z_prev = np.dot(self.mesh.get_transfer_matrix(), X)
        return self.Z_prev

    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        return np.dot(self.mesh.get_transfer_matrix().conj().T, delta)
