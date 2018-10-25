import numpy as np


def is_unitary(m):
    return np.allclose(np.eye(m.shape[0]), m.conj().T @ m, rtol=1e-3, atol=1e-6)
