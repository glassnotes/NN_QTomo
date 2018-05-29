# NN_QTomo

The contents of this folder are an unpublished work in progress, beginning from summer 2017.

Contained are scripts, class files, and a set of running notes which detail
how to do density matrix tomography using a simple feed-forward neural network.

While this technique work for the most part, it scales poorly with system dimension,
and suffers from numerous problems, including state reconstructions that are not positive
semi-definite. Truly, it is a prime example of everything looking like a nail when
you have a large hammer, but it was a very helpful project in terms of learning
about neural networks and how to implement them in Python.

## Requirements

- numpy

## Installation

Simply run
```
python setup.py install
```
in the main directory.

## Contents and usage

You can run this code as:
```
python nn_experiment.py <param_file.txt>
```
This executes a script which generates data for training a neural network (if no data file is available), trains it, and outputs the results either to a log file or to standard output. A sample parameter file is included in the main directory.


The relevant source files are as follows:

`eigvecs.py` : Contains numpy arrays of the MUB vectors (eigenvectors of sets of disjoint
             commuting operators) for various dimensions. Currently included are 2, 3, 4, 8, 32.


`mcexperiment.py` : Monte Carlo simulation of measurement in some or all of the mutually
                  unbiased bases.

`nn_experiment.py` : Eats a parameter file and then generates data (if files not present), trains
                   a neural network, and outputs results to a .log file.

`running_notes.pdf` : A set of running notes and preliminary results.

`sample_params.txt` : Example of a parameter file used by tomodatagenerator and tomoneuralnetwork
                    to generate training data and then run the neural network tomography.

`state_utils.py` : Helper functions for quantum state computations.

`tomodatagenerator.py` : Using multiprocessing to generate Monte Carlo experiment data for a given
                       parameter file. Outputs two CSV file containing the training data (input
                       and output).

`tomoneuralnetwork.py` : Builds and trains a neural network that trains using output files of
                       tomodatagenerator.

Thanks to Roger Melko, Luis Sanchez-Soto, and Ulrich Seyfarth, on who I was able
to bounce many ideas off of while working on this.
