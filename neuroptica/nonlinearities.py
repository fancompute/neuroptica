import autograd.numpy as np
from autograd import jacobian

from neuroptica.settings import NP_COMPLEX


class Nonlinearity:

    def __init__(self, N):
        '''
        Initialize the nonlinearity
        :param N: dimensionality of the nonlinear function
        '''
        self.N = N     # Dimensionality of the nonlinearity
        self.jacobian_re = jacobian(self._forward_pass_re)
        self.jacobian_im = jacobian(self._forward_pass_im)

    def __repr__(self):
        return type(self).__name__

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        '''
        Transform the input fields in the forward direction
        :param X: input fields
        :return: transformed inputs
        '''
        raise NotImplementedError('forward_pass() must be overridden in child class!')

    def _forward_pass_re(self, X: np.ndarray) -> np.ndarray:
        return np.real(self.forward_pass)

    def __repr__(self):
        return type(self).__name__ + '(N={})'.format(self.N)


class ComplexNonlinearity(Nonlinearity):
    '''
    Base class for a complex-valued nonlinearity
    '''

    def __init__(self, N, holomorphic=False, mode="condensed"):
        '''
        Initialize the nonlinearity
        :param N: dimensionality of the nonlinear function
        :param holomorphic: whether the function is holomorphic
        :param mode: for nonholomorphic functions, can be "full", "condensed", or "polar". Full requires that you
        specify 4 derivatives for d{Re,Im}/d{Re,Im}, condensed requires only df/d{Re,Im}, and polar takes Z=re^iphi
        '''
        super().__init__(N)
        self.holomorphic = holomorphic  # Whether the function is holomorphic
        self.mode = mode  # Whether to fully expand to du/da or to use df/da

    def _forward_pass_im(self, X: np.ndarray) -> np.ndarray:
        return np.imag(self.forward_pass)

    def backward_pass(self, gamma: np.ndarray, Z: np.ndarray) -> np.ndarray:
        '''
        Backpropagate a signal through the layer
        :param gamma: backpropagated signal from the (l+1)th layer
        :param Z: output fields from the forward_pass() run
        :return: backpropagated fields delta_l
        '''

        if Z.ndim == 1:
            Z = Z.reshape((Z.size, 1))
            gamma = gamma.reshape((gamma.size, 1))

        n_features, n_samples = Z.shape
        total_derivs = np.zeros(Z.shape, dtype=NP_COMPLEX)

        for i in range(n_samples):
            Z_i = Z[:, i]
            gamma_re, gamma_im = np.real(gamma[:, i]), np.imag(gamma[:, i])
            jac_re = self.jacobian_re(Z_i)
            jac_im = self.jacobian_im(Z_i)

            total_derivs[:, i] = jac_re.T @ gamma_re + jac_im.T @ gamma_im
        return total_derivs

class SPMActivation(Nonlinearity):
    '''
    Lossless SPM activation function

    Parameters
    ---------------
        phase_gain [ rad/(V^2/m^2) ] : The amount of phase shift per unit input "power"
    '''
    def __init__(self, N, gain):
        super().__init__(N)
        self.gain = gain

    def forward_pass(self, Z: np.ndarray):
        gain = self.gain
        phase = gain * np.square(np.abs(Z))
        real_part = np.real(Z) * np.cos(phase) - np.imag(Z) * np.sin(phase)
        imag_part = np.imag(Z) * np.cos(phase) + np.real(Z) * np.sin(phase)

        return real_part + 1j * imag_part


class ElectroOpticActivation(Nonlinearity):
    '''
    Electro-optic activation function with intensity modulation (remod). 

    This activation can be configured either in terms of its physical parameters, detailed
    below, or directly in terms of the feedforward phase gain, g and the biasing phase, phi_b.

    If the electro-optic parameters below are specified g and phi_b are computed for the user.

    Physical parameters and units
    ------------------------------
        alpha: Amount of power tapped off to PD [unitless]
        responsivity: PD responsivity [Watts/amp]
        area: Modal area [micron^2]
        V_pi: Modulator V_pi (voltage required for a pi phase shift) [Volts]
        V_bias: Modulator static bias [Volts]
        R: Transimpedance gain [Ohms]
        impedance: Characteristic impedance for computing optical power [Ohms]
    '''

    def __init__(self, N, alpha=0.1, responsivity=0.8, area=1.0,
    			 V_pi=10.0, V_bias=10.0, R=1e3, impedance=120 * np.pi,
    			 g=None, phi_b=None):

        super().__init__(N)

        self.alpha = alpha

        if g is not None and phi_b is not None:
        	self.g = g
        	self.phi_b = phi_b

        else:
	        # Convert into "feedforward phase gain" and "phase bias" parameters
	        self.g = np.pi * alpha * R * responsivity * area * 1e-12 / 2 / V_pi / impedance
	        self.phi_b  = np.pi * V_bias / V_pi


    def forward_pass(self, Z: np.ndarray):
        alpha, g, phi_b = self.alpha, self.g, self.phi_b
        return 1j * np.sqrt(1-alpha) * np.exp(-1j*0.5*g*np.square(np.abs(Z)) - 1j*0.5*phi_b) * np.cos(0.5*g*np.square(np.abs(Z)) + 0.5*phi_b) * Z


class Abs(Nonlinearity):
    '''
    Represents a transformation z -> |z|. This can be called in any of "full", "condensed", and "polar" modes
    '''

    def __init__(self, N):
        super().__init__(N)

    def forward_pass(self, X: np.ndarray):
        return np.abs(X)

class AbsSquared(Nonlinearity):

    def __init__(self, N):
        super().__init__(N)

    def forward_pass(self, X: np.ndarray):
        return np.abs(X) ** 2

class SoftMax(Nonlinearity):

    def __init__(self, N):
        super().__init__(N)

    def forward_pass(self, X: np.ndarray):
        return np.exp(X) / np.sum(np.exp(X), axis=0)


class LinearMask(Nonlinearity):
    '''Technically not a nonlinearity: apply a linear gain/loss to each element'''

    def __init__(self, N: int, mask=None):
        super().__init__(N)
        if mask is None:
            self.mask = np.ones(N, dtype=NP_COMPLEX)
        else:
            self.mask = np.array(mask, dtype=NP_COMPLEX)

    def forward_pass(self, X: np.ndarray):
        return (X.T * self.mask).T


class bpReLU(Nonlinearity):
    '''
    Discontinuous (but holomorphic and backpropable) ReLU
    f(x_i) = alpha * x_i   if |x_i| <   cutoff
    f(x_i) = x_i           if |x_i| >=   cutoff

    Arguments:
    ----------
        cutoff: value of input |x_i| above which to fully transmit, below which to attentuate
        alpha: attenuation factor f(x_i) = f
    '''
    def __init__(self, N, cutoff=1, alpha=0):
        super().__init__(N)
        self.cutoff = cutoff
        self.alpha = alpha

    def forward_pass(self, X: np.ndarray):
        return (np.abs(X) >= self.cutoff) * X + (np.abs(X) < self.cutoff) * self.alpha * X


class modReLU(Nonlinearity):
    '''
    Contintous, but non-holomorphic and non-simply backpropabable ReLU of the form
    f(z) = (|z| - cutoff) * z / |z| if |z| >= cutoff (else 0)
    see: https://arxiv.org/pdf/1705.09792.pdf  (note, cutoff subtracted in this definition)

    Arguments:
    ----------
        cutoff: value of input |x_i| above which to 
    '''
    def __init__(self, N, cutoff=1):
        super().__init__(N)
        self.cutoff = cutoff

    def forward_pass(self, X: np.ndarray):
        return (np.abs(X) >= self.cutoff) * ( np.abs(X) - self.cutoff ) * X / np.abs(X)


class cReLU(Nonlinearity):
    '''
    Contintous, but non-holomorphic and non-simply backpropabable ReLU of the form
    f(z) = ReLU(Re{z}) + 1j * ReLU(Im{z})
    see: https://arxiv.org/pdf/1705.09792.pdf
    '''
    def __init__(self, N):
        super().__init__(N)

    def forward_pass(self, X: np.ndarray):
        X_re = np.real(X)
        X_im = np.imag(X)
        return (X_re > 0) * X_re + 1j * (X_im > 0) * X_im


class zReLU(Nonlinearity):
    '''
    Contintous, but non-holomorphic and non-simply backpropabable ReLU of the form
    f(z) = z if Re{z} > 0 and Im{z} > 0, else 0
    see: https://arxiv.org/pdf/1705.09792.pdf
    '''
    def __init__(self, N):
        super().__init__(N)

    def forward_pass(self, X: np.ndarray):
        X_re = np.real(X)
        X_im = np.imag(X)
        return (X_re > 0) * (X_im > 0) * X
