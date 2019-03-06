'''The Losses submodule contains classes for computing common loss functions.'''

import numpy as np


class Loss:

    @staticmethod
    def L(X: np.ndarray, T: np.ndarray) -> np.ndarray:
        '''
        The scalar, real-valued loss function (vectorized over multiple X, T inputs)
        :param X: the output of the network
        :param T: the target output
        :return: loss function for each X
        '''
        raise NotImplementedError("Loss function must be specified in child class")

    @staticmethod
    def dL(X: np.ndarray, T: np.ndarray) -> np.ndarray:
        '''
        The derivative of the loss function dL/dX_L used for backpropagation (vectorized over multiple X)
        :param X: the output of the network
        :param T: the target output
        :return: dL/dX_L for each X
        '''
        raise NotImplementedError("Derivative loss function must be specified in child class")


class MeanSquaredError(Loss):

    @staticmethod
    def L(X: np.ndarray, T: np.ndarray) -> np.ndarray:
        return np.sum(1 / 2 * np.abs(T - X) ** 2, axis=0)

    @staticmethod
    def dL(X: np.ndarray, T: np.ndarray) -> np.ndarray:
        return np.conj(X - T)


class CategoricalCrossEntropy(Loss):
    '''Represents categorical cross entropy with a softmax layer implicitly applied to the outputs'''

    @staticmethod
    def L(X: np.ndarray, T: np.ndarray) -> np.ndarray:
        X_softmax = np.exp(X) / np.sum(np.exp(X), axis=0)
        tol = 1e-10
        X_clip = np.clip(X_softmax, tol, 1 - tol)
        return -np.sum(T * np.log(X_clip), axis=0)

    @staticmethod
    def dL(X: np.ndarray, T: np.ndarray) -> np.ndarray:
        X_softmax = np.exp(X) / np.sum(np.exp(X), axis=0)
        return np.conj(X_softmax - T)
