'''[Incomplete module] The initializers submodule includes methods for initializing parameters (such as phase shifter
values) throughout a NetworkLayer.'''

from neuroptica import NetworkLayer


class Initializer:
    '''Base initializer class'''

    def initialize_mesh(self, layer: NetworkLayer):
        '''
        Initialize the phase shifter values for an optical mesh
        :param layer: the NetworkLayer to initialize
        '''
        raise NotImplementedError("initialize_mesh() must be overridden in child class!")


class RandomPhaseInitializer(Initializer):

    def initialize_mesh(self, layer: NetworkLayer):
        pass
