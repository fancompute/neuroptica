'''This module contains a collection of miscellaneous utility functions.'''

import numpy as np
from tqdm import tqdm, tqdm_notebook


def is_unitary(m):
    return np.allclose(np.eye(m.shape[0]), m.conj().T @ m, rtol=1e-3, atol=1e-6)


def is_notebook():
    '''Tests to see if we are running in a jupyter notebook environment'''
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


pbar = tqdm_notebook if is_notebook() else tqdm


def one_hot_to_value(y):
    total = 0
    for i, yi in enumerate(y):
        total += i * yi
    return total / (len(y) - 1)


def generate_ring_planar_dataset(N=400, noise_ratio=0.05, seed=None):
    '''
    Generates a ring of points with one-hot labeling
    :param N: number of points to generate
    :param noise_ratio: multiplier for gaussian noise on radius
    :return: points, labels
    '''
    np.random.seed(seed)
    points = 2 * (np.random.rand(2 * N).reshape((N, 2))) - 1
    labels = np.zeros((N, 2))
    for i, point in enumerate(points):
        if 0.4 <= np.linalg.norm(point) + noise_ratio * np.random.randn() <= 0.8:
            labels[i][0] = 1
        else:
            labels[i][1] = 1
    return points, labels


def generate_diagonal_planar_dataset(N=400, noise_ratio=0.05, seed=None):
    '''
    Generates a ring of points with one-hot labeling
    :param N: number of points to generate
    :param noise_ratio: multiplier for gaussian noise on radius
    :return: points, labels
    '''
    np.random.seed(seed)
    points = 2 * (np.random.rand(2 * N).reshape((N, 2))) - 1
    labels = np.zeros((N, 2))
    for i, pt in enumerate(points):
        if -0.4 <= pt[0] + pt[1] + noise_ratio * np.random.randn() <= 0.4:
            labels[i][0] = 1
        else:
            labels[i][1] = 1
    return points, labels


def generate_separable_planar_dataset(N=100, noise_ratio=0.0, seed=None):
    '''
    Generates a ring of points with one-hot labeling
    :param N: number of points to generate
    :param noise_ratio: multiplier for gaussian noise on radius
    :return: points, labels
    '''
    np.random.seed(seed)
    points = 2 * (np.random.rand(2 * N).reshape((N, 2))) - 1
    labels = np.zeros((N, 2))
    for i, pt in enumerate(points):
        if pt[0] + pt[1] + noise_ratio * np.random.randn() < 0:
            labels[i][0] = 1
        else:
            labels[i][1] = 1
    return points, labels
