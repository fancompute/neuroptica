from typing import Dict, List

import numpy as np

from neuroptica.layers import NetworkLayer


class BaseModel:
    '''
    Base class for all models
    '''

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward_pass(self, d_loss: np.ndarray) -> Dict[str, np.ndarray]:
        raise NotImplementedError


class Model(BaseModel):
    '''
    Functional model class similar to the Keras model class, simulating an optical neural network with multiple layers
    '''

    def __init__(self):
        pass


class Sequential(BaseModel):
    '''
    Feed-foward model class similar to the Keras Sequential() model class
    '''

    def __init__(self, layers: List[NetworkLayer]):
        self.layers = layers
        for i, layer in enumerate(self.layers):
            layer.__name__ = "Layer{}_{}".format(i, layer.__class__.__name__)

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        X_out = X
        for layer in self.layers:
            X_out = layer.forward_pass(X_out)
        return X_out

    def backward_pass(self, d_loss: np.ndarray) -> Dict[str, np.ndarray]:
        '''
        Returns the gradients for each layer resulting from backpropagating from derivative loss function d_loss
        :param d_loss: derivative of the loss function of the outputs
        :return: dictionary of {layer: gradients}
        '''
        backprop_signal = d_loss
        gradients = {}
        for layer in reversed(self.layers):
            backprop_signal = layer.backward_pass(backprop_signal)
            gradients[layer.__name__] = backprop_signal
        return gradients
