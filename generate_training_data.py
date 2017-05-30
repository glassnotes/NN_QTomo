from pynitefields import *
from balthasar import *

import csv

from eigvecs_dim2 import eigenvectors

import qutip as qt
import numpy as np

from pprint import pprint
np.set_printoptions(precision=4, suppress=True)

from scipy.linalg import sqrtm

def generate_data(n_trials, percent_train, f, eigenvectors, bases):
    # Create the MUBs
    mubs = MUBs(f) 
    mle = LBMLE(mubs, eigenvectors)
    mc_engine = LBMLE_MC(eigenvectors)

    op_basis = [x[0][2] for x in mubs.table]

    # Hold the outcomes
    train_in = []
    train_out = []
    
    for i in range(n_trials):
        # Generate a random state with qutip
        state_ket = qt.rand_ket_haar(f.dim)
        state = qt.ket2dm(state_ket).full()

        # Set up the MC engine and generate sample frequencies
        freqs = mc_engine.simulate(bases, state)

        train_in.append(freqs)
        train_out.append([np.trace(np.dot(x, state)) for x in op_basis])
        #train_out.append(state)


    # Split the data set into training and testing
    slice_point = int(n_trials * percent_train)
    test_in = train_in[0:slice_point] 
    test_out = train_out[0:slice_point] 
    train_in = train_in[slice_point:]
    train_out = train_out[slice_point:]

    return train_in, train_out, test_in, test_out
    
