import numpy as np


class ComplexNonlinearity:
    '''
    Base class for a complex-valued nonlinearity
    '''

    def __init__(self, N):
        self.N = N

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
        return gamma * self.df_dZ(Z)

    def df_dZ(self, Z: np.ndarray) -> np.ndarray:
        '''Gives the total complex derivative of the nonlinearity with respect to the input'''
        raise NotImplementedError

    def dRe_dRe(self, Z: np.ndarray) -> np.ndarray:
        '''Gives the derivative of the real part of the nonlienarity w.r.t. the real part of the intput'''
        raise NotImplementedError

    def dRe_dIm(self, Z: np.ndarray) -> np.ndarray:
        '''Gives the derivative of the real part of the nonlienarity w.r.t. the imaginary part of the intput'''
        raise NotImplementedError

    def dIm_dRe(self, Z: np.ndarray) -> np.ndarray:
        '''Gives the derivative of the imaginary part of the nonlienarity w.r.t. the real part of the intput'''
        raise NotImplementedError

    def dIm_dIm(self, Z: np.ndarray) -> np.ndarray:
        '''Gives the derivative of the imaginary part of the nonlienarity w.r.t. the imaginary part of the intput'''
        raise NotImplementedError


class ElectroOpticActivation(ComplexNonlinearity):
    '''
    Ian's electro-optic activation function

    Parameter units
    ---------------
        power_tapped: unitless
        responsivity: watt/amps
        mode_area: um^2
        modulator_voltage: volts
        bias_voltage: volts
        resistance: ohm
        impedence: ohm
    '''

    def __init__(self, N, power_tapped=0.1, responsivity=0.8, mode_area=1.0, modulator_voltage=10.0, bias_voltage=10.0,
                 resistance=1000.0, impedence=120 * np.pi):
        super().__init__(N)
        self.power_tapped = power_tapped
        self.responsivity = responsivity
        self.mode_area = mode_area
        self.modulator_voltage = modulator_voltage
        self.bias_voltage = bias_voltage
        self.resistance = resistance
        self.impedence = impedence

    def forward_pass(self, E_in):
        a, scR, A, Vpi, Vbias, R, Z = self.power_tapped, self.responsivity, self.mode_area, self.modulator_voltage, \
                                      self.bias_voltage, self.resistance, self.impedence
        return E_in / (2 * np.sqrt(2)) * (1 + np.exp(-1j * np.pi * (a * scR * R * A * np.abs(E_in) ** 2) /
                                                     (4 * Z * Vpi)) * np.exp(-1j * np.pi * Vbias / Vpi))

    def backward_pass(self, gamma: np.ndarray, Z: np.ndarray):
        return gamma * self.df_dZ(Z)

    def df_dZ(self, E_in):
        a, scR, A, Vpi, Vbias, R, Z = self.power_tapped, self.responsivity, self.mode_area, self.modulator_voltage, \
                                      self.bias_voltage, self.resistance, self.impedence
        abs_prime_E_in = 1
        return 1 / (2 * np.sqrt(2)) \
               + (np.exp(-1j * np.pi * (Vbias / Vpi + np.abs(E_in) ** 2 * A * R * scR / (4e12 * Vpi * Z)))
                  * 2e12 * Vpi * Z - 1j * np.pi * A * E_in * R * scR * np.abs(E_in) * abs_prime_E_in
                  / 4e12 * np.sqrt(2) * Vpi * Z)


class Abs(ComplexNonlinearity):

    def forward_pass(self, X: np.ndarray):
        return np.abs(X)

    def df_dZ(self, Z):
        return np.ones(Z.shape)


class AbsSquared(ComplexNonlinearity):

    def forward_pass(self, X: np.ndarray):
        return np.abs(X) ** 2

    def df_dZ(self, Z):
        return 2 * np.abs(Z)
