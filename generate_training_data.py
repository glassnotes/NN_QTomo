from pynitefields import *
from balthasar import *

import csv

import qutip as qt
import numpy as np

from pprint import pprint
np.set_printoptions(precision=4, suppress=True)

op_basis = [np.array([[0, 1], [1, 0]]),
            np.array([[0, -1j], [1j, 0]]),
            np.array([[1, 0], [0, -1]])]

def generate_data(n_trials, percent_train, f, eigenvectors, bases, with_lbmle = False):
    """ Generate training/testing data sets for NN, and also store the 
        testing data separately so we can compare it with the LBMLE reconstruction
        algorithm.
    """
    # Create the MUBs and MLE simulator
    mubs = MUBs(f) 
    mc_engine = LBMLE_MC(eigenvectors)

    # Hold the outcomes
    train_in = []
    train_out = []
    lbmle_freqs = []
    
    for i in range(n_trials):
        # Generate a random state with qutip
        state_ket = qt.rand_ket_haar(f.dim)
        state = qt.ket2dm(state_ket).full()

        freqs = mc_engine.simulate(bases, state)

        # Add to lbmle data set
        if with_lbmle:
            lbmle_freqs.append(freqs)

        # Flatten and add to NN data set
        flat_freqs = []
        for s in freqs:
            flat_freqs.extend(s)

        # Compute the basis coefficients; they should be real, but sometimes
        # there is a very very tiny (1e-17) complex part, so throw that away
        coefs = [np.trace(np.dot(x, state)).real for x in op_basis]

        train_in.append(flat_freqs)
        train_out.append(coefs)

    # Split the data set into training and testing
    slice_point = int(n_trials * percent_train)
    test_in = np.array(train_in[0:slice_point])
    test_out = np.array(train_out[0:slice_point])
    train_in = np.array(train_in[slice_point:])
    train_out = np.array(train_out[slice_point:])

    if with_lbmle: 
        return train_in, train_out, test_in, test_out, lbmle_freqs[0:slice_point] 

    # No LBMLE data, just return training/testing data
    return train_in, train_out, test_in, test_out
    
