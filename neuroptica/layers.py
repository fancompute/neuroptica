class NetworkLayer:
	'''
	Represents a logical layer in a neural network (different from ComponentLayer)
	'''

	def forward_pass(self, X):
		raise NotImplementedError('forward_pass() must be overridden in child class!')
