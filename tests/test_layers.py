import unittest

import numpy as np

from neuroptica.layers import ClementsLayer
from neuroptica.optimizers import Optimizer
from tests.base import NeuropticaTest


class TestLayers(NeuropticaTest):
	'''Tests for Network layers'''

	def test_ClementsLayer(self):
		'''Tests the Clements layer'''
		for N in [8, 9]:
			c = ClementsLayer(N)
			X = np.random.rand(N)
			X_out = c.forward_pass(X)
			# Check that a unitary transformation was done
			self.assert_allclose(np.linalg.norm(X), np.linalg.norm(X_out))

	def test_ClementsLayerOptimization(self):
		'''Tests the Clements layer optimization'''
		for N in [8, 9]:
			c = ClementsLayer(N)
			o = Optimizer(c.mesh)

			X_in = np.random.rand(N)
			X_out = c.forward_pass(X_in)

			fields = o.compute_phase_shifter_fields(X_in, align="right")
			adjoint_fields = o.compute_adjoint_phase_shifter_fields(X_out, align="right")

			# Check that a unitary transformation was done
			for layer_fields in fields:
				for component_fields in layer_fields:
					self.assert_allclose(np.linalg.norm(X_in), np.linalg.norm(component_fields))

			# Check results match at end
			output_fields = fields[-1][-1]
			self.assert_allclose(X_out, output_fields)

			# Check that a unitary transformation was done
			for layer_fields_adj in adjoint_fields:
				for component_fields_adj in layer_fields_adj:
					self.assert_allclose(np.linalg.norm(X_in), np.linalg.norm(component_fields_adj))

			# Check results match at end
			output_fields_adj = o.compute_adjoint_phase_shifter_fields(X_out, align="left")[-1][-1]
			self.assert_allclose(output_fields_adj, X_in)

			# Check that adjoint field of X_out equals regular field of X_in
			for layer_fields, layer_fields_adj in zip(fields, reversed(adjoint_fields)):
				for component_fields, component_fields_adj in zip(layer_fields, reversed(layer_fields_adj)):
					self.assert_allclose(component_fields, component_fields_adj)


if __name__ == "__main__":
	unittest.main()
