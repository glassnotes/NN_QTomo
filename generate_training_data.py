from pynitefields import *
from balthasar import *

import csv

import qutip as qt
import numpy as np

from pprint import pprint
np.set_printoptions(precision=4, suppress=True)

def generate_data(n_trials, percent_train, f, eigenvectors, bases):
    # Create the MUBs
    mubs = MUBs(f) 
    #mle = LBMLE(mubs, eigenvectors)
    #mc_engine = LBMLE_MC(eigenvectors)

    op_basis = [x[0][2] for x in mubs.table]

    # Hold the outcomes
    train_in = []
    train_out = []
    
    for i in range(n_trials):
        # Generate a random state with qutip
        state_ket = qt.rand_ket_haar(f.dim)
        state = qt.ket2dm(state_ket).full()

        # Set up the MC engine and generate sample frequencies
        #freqs = mc_engine.simulate(bases, state)

        # For now just generate random frequencies and make sure they
        # sum to 1.
        freqs = np.random.rand(len(bases), f.dim) 
        for j in range(len(freqs)):
            freqs[j][1] = 1 - freqs[j][0] 

        flat_freqs = []
        for s in freqs:
            flat_freqs.extend(s)

        train_in.append(flat_freqs)
        train_out.append([2 * np.trace(np.dot(x, state)) for x in op_basis])
        #train_out.append(state)


    # Split the data set into training and testing
    slice_point = int(n_trials * percent_train)
    test_in = np.array(train_in[0:slice_point])
    test_out = np.array(train_out[0:slice_point])
    train_in = np.array(train_in[slice_point:])
    train_out = np.array(train_out[slice_point:])

    return train_in, train_out, test_in, test_out
    
