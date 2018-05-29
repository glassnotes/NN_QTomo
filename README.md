# NN_QTomo
My playground for using neural networks to perform density matrix quantum tomography.

The code in this repository is a work-in-progress that was begun in summer 2017. It has not been published, as the conclusions I came to were that while using a neural network to do tomography works, it is a prime example of everything looking like a nail when we have a really big hammer.

In particular, higher-dimensional systems (unsurprisingly) require far too much data and training time to be practical. It also suffers from an issue where higher-dimensional (i.e. > 2) reconstructions are not positive semidefinite, though they are very, very close. Finally, given the previous points, I did not come up with anything good for mixed states reconstruction.

Nevertheless, if you are interested in any kind of collaboration to pursue this further, please don't hesitate to contact me at odimatte@uwaterloo.ca

# Problem description

More details can be found in the notes document contained within.

The general idea is that we perform supervised learning where the input data is a frequencies of measurement outcomes in mutually unbiased bases, and the output is the parameters of the (generalized) Bloch vector.

# Requirements

- numpy

# Installation

Install by performing
```
python setup.py install
```
in the main directory.

The script `nn_experiment.py` can be run from command line using
```
python nn_experiment.py <paramfile>
```
A sample parameter file is provided in the main directory.
