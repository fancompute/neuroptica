'''This module contains a collection of physical and aphysical activation functions. Nonlinearities can be incorporated
into an optical neural network by using the Activation(nonlinearity) NetworkLayer.'''


import numpy as np

from neuroptica.settings import NP_COMPLEX


class Nonlinearity:

    def __init__(self, N):
        '''
        Initialize the nonlinearity
        :param N: dimensionality of the nonlinear function
        '''
        self.N = N  # Dimensionality of the nonlinearity

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        '''
        Transform the input fields in the forward direction
        :param X: input fields
        :return: transformed inputs
        '''
        raise NotImplementedError('forward_pass() must be overridden in child class!')

    def backward_pass(self, gamma: np.ndarray, Z: np.ndarray) -> np.ndarray:
        '''
        Backpropagate a signal through the layer
        :param gamma: backpropagated signal from the (l+1)th layer
        :param Z: output fields from the forward_pass() run
        :return: backpropagated fields delta_l
        '''
        raise NotImplementedError('backward_pass() must be overridden in child class!')

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

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        '''
        Transform the input fields in the forward direction
        :param X: input fields
        :return: transformed inputs
        '''
        raise NotImplementedError('forward_pass() must be overridden in child class!')

    def backward_pass(self, gamma: np.ndarray, Z: np.ndarray) -> np.ndarray:
        '''
        Backpropagate a signal through the layer
        :param gamma: backpropagated signal from the (l+1)th layer
        :param Z: output fields from the forward_pass() run
        :return: backpropagated fields delta_l
        '''
        # raise NotImplementedError('backward_pass() must be overridden in child class!')
        if self.holomorphic:
            return gamma * self.df_dZ(Z)

        else:

            if self.mode == "full":
                a, b = np.real(Z), np.imag(Z)
                return np.real(gamma) * (self.dRe_dRe(a, b) - 1j * self.dRe_dIm(a, b)) + \
                       np.imag(gamma) * (-1 * self.dIm_dRe(a, b) + 1j * self.dIm_dIm(a, b))

            elif self.mode == "condensed":
                a, b = np.real(Z), np.imag(Z)
                return np.real(gamma * self.df_dRe(a, b)) - 1j * np.real(gamma * self.df_dIm(a, b))

            elif self.mode == "polar":
                r, phi = np.abs(Z), np.angle(Z)
                return np.exp(-1j * phi) * \
                       (np.real(gamma * self.df_dr(r, phi)) - 1j / r * np.real(gamma * self.df_dphi(r, phi)))

    def df_dZ(self, Z: np.ndarray) -> np.ndarray:
        '''Gives the total complex derivative of the (holomorphic) nonlinearity with respect to the input'''
        raise NotImplementedError

    def df_dRe(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        '''Gives the derivative of the nonlinearity with respect to the real part alpha of the input'''
        raise NotImplementedError

    def df_dIm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        '''Gives the derivative of the nonlinearity with respect to the imaginary part beta of the input'''
        raise NotImplementedError

    def dRe_dRe(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        '''Gives the derivative of the real part of the nonlienarity w.r.t. the real part of the input'''
        raise NotImplementedError

    def dRe_dIm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        '''Gives the derivative of the real part of the nonlienarity w.r.t. the imaginary part of the input'''
        raise NotImplementedError

    def dIm_dRe(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        '''Gives the derivative of the imaginary part of the nonlienarity w.r.t. the real part of the input'''
        raise NotImplementedError

    def dIm_dIm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        '''Gives the derivative of the imaginary part of the nonlienarity w.r.t. the imaginary part of the input'''
        raise NotImplementedError

    def df_dr(self, r: np.ndarray, phi: np.ndarray) -> np.ndarray:
        '''Gives the derivative of the nonlinearity with respect to the magnitude r of the input'''
        raise NotImplementedError

    def df_dphi(self, r: np.ndarray, phi: np.ndarray) -> np.ndarray:
        '''Gives the derivative of the nonlinearity with respect to the angle phi of the input'''
        raise NotImplementedError


class SPMActivation(ComplexNonlinearity):
    '''
    Lossless SPM activation function

    Parameters
    ---------------
        phase_gain [ rad/(V^2/m^2) ] : The amount of phase shift per unit input "power"
    '''

    def __init__(self, N, gain):
        super().__init__(N, mode="condensed")
        self.gain = gain

    def forward_pass(self, Z: np.ndarray):
        gain = self.gain
        return Z * np.exp(-1j * gain * np.square(np.abs(Z)))

    def df_dRe(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        gain = self.gain
        Z = a + 1j * b
        return np.exp(-1j * gain * np.square(np.abs(Z))) * (-2j * np.square(a) * gain + 2 * a * b * gain + 1)

    def df_dIm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        gain = self.gain
        Z = a + 1j * b
        return np.exp(-1j * gain * np.square(np.abs(Z))) * (-2j * a * b * gain + 2 * np.square(b) * gain + 1j)


class ElectroOpticActivation(ComplexNonlinearity):
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

        super().__init__(N, mode="condensed")

        self.alpha = alpha

        if g is not None and phi_b is not None:
            self.g = g
            self.phi_b = phi_b

        else:
            # Convert into "feedforward phase gain" and "phase bias" parameters
            self.g = np.pi * alpha * R * responsivity * area * 1e-12 / 2 / V_pi / impedance
            self.phi_b = np.pi * V_bias / V_pi

    def forward_pass(self, Z: np.ndarray):
        alpha, g, phi_b = self.alpha, self.g, self.phi_b
        return 1j * np.sqrt(1 - alpha) * np.exp(-1j * 0.5 * g * np.square(np.abs(Z)) - 1j * 0.5 * phi_b) * np.cos(
            0.5 * g * np.square(np.abs(Z)) + 0.5 * phi_b) * Z

    def df_dRe(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        alpha, g, phi_b = self.alpha, self.g, self.phi_b
        return np.sqrt(1 - alpha) * np.exp((-0.5 * 1j) * g * (a - 1j * b) * (a + 1j * b) - (0.5 * 1j) * phi_b) * (
                a * g * (b - 1j * a) * np.sin(0.5 * a ** 2 * g + 0.5 * b ** 2 * g + 0.5 * phi_b) + (
                a ** 2 * g + 1j * a * b * g + 1j) * np.cos(0.5 * a ** 2 * g + 0.5 * b ** 2 * g + 0.5 * phi_b))

    def df_dIm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        alpha, g, phi_b = self.alpha, self.g, self.phi_b
        return np.sqrt(1 - alpha) * np.exp((-0.5 * 1j) * g * (a - 1j * b) * (a + 1j * b) - (0.5 * 1j) * phi_b) * (
                b * g * (b - 1j * a) * np.sin(0.5 * a ** 2 * g + 0.5 * b ** 2 * g + 0.5 * phi_b) + (
                a * b * g + 1j * b ** 2 * g - 1) * np.cos(0.5 * a ** 2 * g + 0.5 * b ** 2 * g + 0.5 * phi_b))


class Abs(ComplexNonlinearity):
    '''
    Represents a transformation z -> |z|. This can be called in any of "full", "condensed", and "polar" modes
    '''

    def __init__(self, N, mode="polar"):
        super().__init__(N, holomorphic=False, mode=mode)

    def forward_pass(self, X: np.ndarray):
        return np.abs(X)

    def dRe_dRe(self, a: np.ndarray, b: np.ndarray):
        return a / np.sqrt(a ** 2 + b ** 2)

    def dRe_dIm(self, a: np.ndarray, b: np.ndarray):
        return b / np.sqrt(a ** 2 + b ** 2)

    def dIm_dRe(self, a: np.ndarray, b: np.ndarray):
        return 0 * a

    def dIm_dIm(self, a: np.ndarray, b: np.ndarray):
        return 0 * b

    def df_dRe(self, a: np.ndarray, b: np.ndarray):
        return a / np.sqrt(a ** 2 + b ** 2)

    def df_dIm(self, a: np.ndarray, b: np.ndarray):
        return b / np.sqrt(a ** 2 + b ** 2)

    def df_dr(self, r: np.ndarray, phi: np.ndarray):
        return np.ones(r.shape, dtype=NP_COMPLEX)

    def df_dphi(self, r: np.ndarray, phi: np.ndarray):
        return 0 * phi


class AbsSquared(ComplexNonlinearity):
    '''Maps z -> |z|^2, corresponding to power measurement by a photodetector.'''

    def __init__(self, N):
        super().__init__(N, holomorphic=False, mode="polar")

    def forward_pass(self, X: np.ndarray):
        return np.abs(X) ** 2

    def df_dr(self, r: np.ndarray, phi: np.ndarray):
        return 2 * r

    def df_dphi(self, r: np.ndarray, phi: np.ndarray):
        return 0 * phi


class Sigmoid(Nonlinearity):
    '''Sigmoid activation; maps z -> 1 / (1 + np.exp(-z))'''

    def forward_pass(self, X: np.ndarray):
        return 1 / (1 + np.exp(-X))

    def backward_pass(self, gamma: np.ndarray, Z: np.ndarray):
        sigma = 1 / (1 + np.exp(-Z))
        return sigma * (1 - sigma) * gamma


class SoftMax(Nonlinearity):
    '''Applies softmax to the inputs. Do not use in with categorical cross entropy, which implicitly includes this.'''

    def forward_pass(self, X: np.ndarray):
        return np.exp(X) / np.sum(np.exp(X), axis=0)

    def backward_pass(self, gamma: np.ndarray, Z: np.ndarray):
        softmax = np.exp(Z) / np.sum(np.exp(Z), axis=0)

        n_features, n_samples = Z.shape
        total_derivs = np.zeros(Z.shape, dtype=NP_COMPLEX)

        for i in range(n_samples):
            s = softmax[:, i].reshape(-1, 1)
            jac = np.diagflat(s) - np.dot(s, s.T)
            total_derivs[:, i] = jac.T @ gamma[:, i]

        return total_derivs


class LinearMask(ComplexNonlinearity):
    '''Technically not a nonlinearity: apply a linear gain/loss to each element'''

    def __init__(self, N: int, mask=None):
        super().__init__(N, holomorphic=True)
        if mask is None:
            self.mask = np.ones(N, dtype=NP_COMPLEX)
        else:
            self.mask = np.array(mask, dtype=NP_COMPLEX)

    def forward_pass(self, X: np.ndarray):
        return (X.T * self.mask).T

    def df_dZ(self, Z: np.ndarray):
        z_broadcaster = np.ones(Z.shape)
        return (z_broadcaster.T * self.mask).T
        # return ((Z.T * self.mask) / Z.T).T


class bpReLU(ComplexNonlinearity):
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
        super().__init__(N, holomorphic=True)
        self.cutoff = cutoff
        self.alpha = alpha

    def forward_pass(self, X: np.ndarray):
        return (np.abs(X) >= self.cutoff) * X + (np.abs(X) < self.cutoff) * self.alpha * X

    def df_dZ(self, Z: np.ndarray):
        return (np.abs(Z) >= self.cutoff) * 1 + (np.abs(Z) < self.cutoff) * self.alpha * 1


class modReLU(ComplexNonlinearity):
    '''
    Contintous, but non-holomorphic and non-simply backpropabable ReLU of the form
    f(z) = (|z| - cutoff) * z / |z| if |z| >= cutoff (else 0)
    see: https://arxiv.org/pdf/1705.09792.pdf  (note, cutoff subtracted in this definition)

    Arguments:
    ----------
        cutoff: value of input |x_i| above which to
    '''

    def __init__(self, N, cutoff=1):
        super().__init__(N, holomorphic=False, mode="polar")
        self.cutoff = cutoff

    def forward_pass(self, X: np.ndarray):
        return (np.abs(X) >= self.cutoff) * (np.abs(X) - self.cutoff) * X / np.abs(X)

    def df_dr(self, r: np.ndarray, phi: np.ndarray):
        return (r >= self.cutoff) * np.exp(1j * phi)

    def df_dphi(self, r: np.ndarray, phi: np.ndarray):
        return (r >= self.cutoff) * 1j * (r - self.cutoff) * np.exp(1j * phi)


class cReLU(ComplexNonlinearity):
    '''
    Contintous, but non-holomorphic and non-simply backpropabable ReLU of the form
    f(z) = ReLU(Re{z}) + 1j * ReLU(Im{z})
    see: https://arxiv.org/pdf/1705.09792.pdf
    '''

    def __init__(self, N):
        super().__init__(N, holomorphic=False, mode="condensed")

    def forward_pass(self, X: np.ndarray):
        X_re = np.real(X)
        X_im = np.imag(X)
        return (X_re > 0) * X_re + 1j * (X_im > 0) * X_im

    def df_dRe(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a > 0)

    def df_dIm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return 1j * (b > 0)


class zReLU(ComplexNonlinearity):
    '''
    Contintous, but non-holomorphic and non-simply backpropabable ReLU of the form
    f(z) = z if Re{z} > 0 and Im{z} > 0, else 0
    see: https://arxiv.org/pdf/1705.09792.pdf
    '''

    def __init__(self, N):
        super().__init__(N, holomorphic=False, mode="condensed")

    def forward_pass(self, X: np.ndarray):
        X_re = np.real(X)
        X_im = np.imag(X)
        return (X_re > 0) * (X_im > 0) * X

    def df_dRe(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a > 0) * (b > 0)

    def df_dIm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a > 0) * (b > 0) * 1j
