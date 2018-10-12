import numpy as np
from numpy import pi
from scipy.optimize import fsolve
from tqdm import tqdm_notebook as pbar

__all__ = ['clementsDecomposition']


def T(i, j, N, theta, phi):
	matrix = np.eye(N, dtype=np.complex64)
	matrix[i][i] = np.exp(1j * phi) * np.cos(theta)
	matrix[i][j] = -1 * np.sin(theta)
	matrix[j][i] = np.exp(1j * phi) * np.sin(theta)
	matrix[j][j] = np.cos(theta)
	return matrix


def _TmnU_mx(theta_phi, Umx, Unx):
	theta, phi = theta_phi
	target = np.exp(1j * phi) * np.cos(theta) * Umx - np.sin(theta) * Unx
	return target.real, target.imag


def _TmnU_nx(theta_phi, Umx, Unx):
	theta, phi = theta_phi
	target = np.exp(1j * phi) * np.sin(theta) * Umx + np.cos(theta) * Unx
	return target.real, target.imag


def leftNullify(U, m, n, targetRow, targetCol):
	'''
	Finds a matrix Tmn to nullify an entry at Urow, Ucol of (Tmn U)
	'''
	assert m < n, "Requires m<n!"
	assert targetRow != targetCol, "Can't nullify diagonal!"
	Umx, Unx = U[m][targetCol], U[n][targetCol]
	if targetRow == m:
		theta, phi = fsolve(_TmnU_mx, (pi * np.random.rand(), 2 * pi * np.random.rand()), args=(Umx, Unx))
	elif targetRow == n:
		theta, phi = fsolve(_TmnU_nx, (pi * np.random.rand(), 2 * pi * np.random.rand()), args=(Umx, Unx))
	else:
		raise ValueError("Nullification target entry must be in row m or n.")
	# Constrain theta, phi and check results
	theta = np.mod(theta, 2 * pi)  # theta ranges from [0,pi)
	phi = np.mod(phi, 2 * pi)  # phi ranges from [0, 2pi)
	Tmn = T(m, n, U.shape[0], theta, phi)
	cache = {"m": m, "n": n, "theta": theta, "phi": phi}
	#     assert np.isclose(0, (Tmn @ U)[targetRow][targetCol], rtol = 1e-1, atol = 1e-2), "Fail!"
	return Tmn, cache


def Tinv(i, j, N, theta, phi):
	return T(i, j, N, theta, phi).conj().T


def _UTmnInv_xm(theta_phi, Uxm, Uxn):
	theta, phi = theta_phi
	target = Uxm * np.exp(-1j * phi) * np.cos(theta) - Uxn * np.sin(theta)
	return target.real, target.imag


def _UTmnInv_xn(theta_phi, Uxm, Uxn):
	theta, phi = theta_phi
	target = Uxm * np.exp(-1j * phi) * np.sin(theta) + Uxn * np.cos(theta)
	return target.real, target.imag


def rightNullify(U, m, n, targetRow, targetCol):
	'''
	Finds a matrix Tmn^-1 to nullify an entry at Urow, Ucol of (U Tmn^-1)
	'''
	assert m < n, "Requires m<n!"
	assert targetRow != targetCol, "Can't nullify diagonal!"
	Uxm, Uxn = U[targetRow][m], U[targetRow][n]
	if targetCol == m:
		theta, phi = fsolve(_UTmnInv_xm, (pi * np.random.rand(), 2 * pi * np.random.rand()), args=(Uxm, Uxn))
	elif targetCol == n:
		theta, phi = fsolve(_UTmnInv_xn, (pi * np.random.rand(), 2 * pi * np.random.rand()), args=(Uxm, Uxn))
	else:
		raise ValueError("Nullification target entry must be in column m or n.")
	# Constrain theta, phi and check results
	theta = np.mod(theta, 2 * pi)  # theta ranges from [0,pi)
	phi = np.mod(phi, 2 * pi)  # phi ranges from [0, 2pi)
	TmnInv = Tinv(m, n, U.shape[0], theta, phi)
	cache = {"m": m, "n": n, "theta": theta, "phi": phi}
	#     assert np.isclose(0, (U @ TmnInv)[targetRow][targetCol], rtol = 1e-1, atol = 1e-2), "Fail!"
	return TmnInv, cache


def clementsDecomposition(U, verbose=False, show_progress=False):
	'''
	Performs a Clements decomposition on a unitary matrix U
	'''
	N = U.shape[0]
	Uhat = np.copy(U)
	Tlist = []
	Tcaches = []
	TinvList = []
	TinvCaches = []
	count = 0
	if show_progress:
		iterator = pbar(range(1, N))
	else:
		iterator = range(1, N)
	for i in iterator:  # for i = 1 to N-1
		if i % 2 != 0:  # if i is odd
			for j in range(i):  # for j = 0 to i-1
				# Find a T_i-j,i-j+1^-1 matrix to nullify (N-j, i-j) of U
				m, n = i - j, i - j + 1
				targetRow, targetCol = N - j, i - j
				TmnInv, cache = rightNullify(Uhat, m - 1, n - 1, targetRow - 1, targetCol - 1)  # 0-indexed
				if verbose:
					print("count={}, i={}, j={}, m={}, n={}, row={}, col={}\n".format(count, i, j, m - 1, n - 1,
																					  targetRow - 1, targetCol - 1))
					print(Uhat, "\n\n", TmnInv, "\n\n", Uhat @ TmnInv, "\n\n\n\n")
				Uhat = Uhat @ TmnInv
				TinvList.append(TmnInv)
				TinvCaches.append(cache)
				count += 1
		else:  # if i is even
			for j in range(1, i + 1):  # for j = 1 to i
				# Find a T_n+j-i-1,N+j-i matrix to nullify (N+j-i, j) of U
				m, n = N + j - i - 1, N + j - i
				targetRow, targetCol = N + j - i, j
				Tmn, cache = leftNullify(Uhat, m - 1, n - 1, targetRow - 1, targetCol - 1)  # 0-indexed
				if verbose:
					print("count={}, i={}, j={}, m={}, n={}, row={}, col={}\n".format(count, i, j, m - 1, n - 1,
																					  targetRow - 1, targetCol - 1))
					print(Uhat, "\n\n", Tmn, "\n\n", Tmn @ Uhat, "\n\n\n\n")
				Uhat = Tmn @ Uhat
				Tlist.insert(0, Tmn)
				Tcaches.insert(0, cache)
				count += 1
	# Now we have Uhat = Tlist @ U @ TinvList with Uhat diagonal
	Tlist = [T.conj().T for T in reversed(Tlist)]
	TinvList = [Tinv.conj().T for Tinv in reversed(TinvList)]
	Tcaches.reverse()
	TinvCaches.reverse()
	Tlist, D, TinvList = TinvList, Uhat, Tlist  # Switch names for consistency - Tlist is physical
	Tcaches, TinvCaches = TinvCaches, Tcaches
	# Now U = TinvList @ D @ Tlist
	return TinvList, D, Tlist, TinvCaches, Tcaches


def clementsNormalForm(U, verbose=False):
	'''
	Expresses U as U = D * Prod(T_mn) for D = diagonal phase shift block
	'''
	# TinvList, D, Tlist, TinvCaches, Tcaches = clementsDecomposition(U, verbose = verbose)
	#     Dprime = np.copy(D)
	# Iteratively move D all the way to the left
	# for Tinv, TinvCache in reversed(zip(TinvList, TinvCaches)):
	#     #         Dprime = Tinv @ Dprime @ Tinv
	#     DTmn = np.linalg.tensorsolve(Tinv.conj().T, D)
	#     print("Dprime = \n{}\n".format(Dprime))
	#     Tlist.insert(0, Tinv.conj().T)
	assert False, "Not finished!"
# return Dprime, Tlist
