# neuroptica [![Documentation Status](https://readthedocs.org/projects/neuroptica/badge/?version=latest)](https://neuroptica.readthedocs.io/en/latest/?badge=latest) [![Build Status](https://travis-ci.com/fancompute/neuroptica.svg?token=CSoUuvqmixfJpdwkLqet&branch=master)](https://travis-ci.com/fancompute/neuroptica)

`neuroptica` is a flexible chip-level simulation platform for nanophotonic neural networks written in Python/NumPy. It provides a wide range of abstracton levels for simulating optical NN's: the lowest-level functionality allows you to manipulate the arrangement and properties of individual phase shifters on a simulated chip, and the highest-level features provide a Keras-like API for designing optical NN by stacking network layers.


## Installation

The easiest way to get started with `neuroptica` is to install directly from the Python package manager:

```
pip install neuroptica
```

Alternately, you can clone the repository source code and edit it as needed with 

```
git clone https://github.com/fancompute/neuroptica.git
```

and in your program or notebook, add

```python
import sys
sys.path.append('path/to/neuroptica')
``` 


## Getting started

For an overview of `neuroptica`, read the [documentation](https://neuroptica.readthedocs.io). Example notebooks are included in the [`neuroptica-notebooks`](https://github.com/fancompute/neuroptica-notebooks) repository:

- [Planar data classification using electro-optic activation functions](https://github.com/fancompute/neuroptica-notebooks/blob/master/neuroptica_demo.ipynb)

![neuroptica training demo](https://raw.githubusercontent.com/fancompute/neuroptica/master/img/neuroptica_demo.gif)


## Authors

`neuroptica` was written by [Ben Bartlett](https://github.com/bencbartlett), [Momchil Minkov](https://github.com/momchilmm), [Tyler Hughes](https://github.com/twhughes), and  [Ian Williamson](https://github.com/ianwilliamson).
