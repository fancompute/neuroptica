import numpy as np

from neuroptica.components.component_layers import MZILayer, OpticalMesh, PhaseShifterLayer
from neuroptica.nonlinearities import ComplexNonlinearity
from neuroptica.settings import NP_COMPLEX


class NetworkLayer:
    '''
    Represents a logical layer in a neural network (different from ComponentLayer)
    '''

    def __init__(self, input_size: int, output_size: int, initializer=None):
        self.input_size = input_size
        self.output_size = output_size
        self.initializer = initializer
        self.input_prev: np.ndarray = None
        self.output_prev: np.ndarray = None

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError('forward_pass() must be overridden in child class!')

    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        raise NotImplementedError('backward_pass() must be overridden in child class!')


class DropMask(NetworkLayer):
    '''Drop specified ports entirely'''

    def __init__(self, N: int, keep_ports=None, drop_ports=None):
        if (keep_ports is not None and drop_ports is not None) or (keep_ports is None and drop_ports is None):
            raise ValueError("specify exactly one of keep_ports or drop_ports")
        if keep_ports:
            if isinstance(keep_ports, range):
                keep_ports = list(keep_ports)
            elif isinstance(keep_ports, int):
                keep_ports = [keep_ports]
            self.ports = keep_ports
        elif drop_ports:
            ports = list(range(N))
            for port in drop_ports:
                ports.remove(port)
            self.ports = ports
        super().__init__(N, len(self.ports))

    def forward_pass(self, X: np.ndarray):
        # self.input_prev = X
        # self.output_prev = X[self.ports, :]
        # return self.output_prev
        return X[self.ports]

    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        n_features, n_samples = delta.shape
        delta_back = np.zeros((self.input_size, n_samples), dtype=NP_COMPLEX)
        for i in range(n_features):
            delta_back[self.ports[i]] = delta[i]
        return delta_back


class Activation(NetworkLayer):
    '''
    Represents a (nonlinear) activation layer. Note that in this layer, the usage of X and Z are reversed! (Z is input,
    X is output, input for next linear layer)
    '''

    def __init__(self, nonlinearity: ComplexNonlinearity):
        super().__init__(nonlinearity.N, nonlinearity.N)
        self.nonlinearity = nonlinearity

    def forward_pass(self, Z: np.ndarray) -> np.ndarray:
        self.input_prev = Z
        self.output_prev = self.nonlinearity.forward_pass(Z)
        return self.output_prev

    def backward_pass(self, gamma: np.ndarray) -> np.ndarray:
        return self.nonlinearity.backward_pass(gamma, self.input_prev)


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

    def forward_pass(self, X: np.ndarray, field_store=False, partial_vectors=False) -> np.ndarray:
        self.input_prev = X
        if not field_store:
            self.output_prev = np.dot(self.mesh.get_transfer_matrix(), X)
        else:
            self.mesh.forward_fields = self.mesh.compute_phase_shifter_fields(X, 
                                                align="right", partial_vectors=partial_vectors)
            self.output_prev = np.copy(self.mesh.forward_fields[-1][-1])

        return self.output_prev

    def backward_pass(self, delta: np.ndarray, field_store=False, partial_vectors=False) -> np.ndarray:        
        if not field_store:
            return np.dot(self.mesh.get_transfer_matrix().T, delta)
        else:
            self.mesh.adjoint_fields = self.mesh.compute_adjoint_phase_shifter_fields(delta, 
                                                align="right", partial_vectors=partial_vectors)
            if isinstance(self.mesh.layers[0], PhaseShifterLayer):
                output_back = np.dot(self.mesh.layers[0].get_transfer_matrix().T, self.mesh.adjoint_fields[-1][-1])
                return output_back
            else:
                ValueError("Field_store will not work in this case, please set to False")


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

        for start, end in zip(mzi_limits_lower, mzi_limits_upper):
            layers.append(MZILayer.from_waveguide_indices(N, list(range(start, end + 1))))

        self.mesh = OpticalMesh(N, layers)

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        self.input_prev = X
        self.output_prev = np.dot(self.mesh.get_transfer_matrix(), X)
        return self.output_prev

    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        return np.dot(self.mesh.get_transfer_matrix().T, delta)
