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
