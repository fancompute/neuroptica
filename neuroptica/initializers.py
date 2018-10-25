from neuroptica import NetworkLayer


class Initializer:
	'''Base initializer class'''

	def initialize_mesh(self, layer: NetworkLayer):
		'''Initialize the phase shifter values for an optical mesh'''
		raise NotImplementedError("initialize_mesh() must be overridden in child class!")


class RandomPhaseInitializer(Initializer):

	def initialize_mesh(self, layer: NetworkLayer):
		pass
