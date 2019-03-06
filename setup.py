import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neuroptica",
    version="0.1.0",
    author="Ben Bartlett",
    author_email="benbartlett@stanford.edu",
    description="Nanophotonic Neural Network Simulator",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fancompute/neuroptica",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
    install_requires=[
        "numpy",
        "scipy",
        "numba",
        "tqdm"
    ],
)
