from pynitefields import *
from balthasar import *

import csv

import qutip as qt
import numpy as np

from pprint import pprint
np.set_printoptions(precision=4, suppress=True)

from math import sqrt

I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

sqp = [I, X, Y, Z]

# Dim 2 case
#op_basis = [X, Y, Z]

# Dim 3 case - Gell-Mann matrces as our basis, see Bertlmann and Krammer
ls12 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
ls13 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
ls23 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
la12 = np.array([[0,-1j, 0], [1j, 0, 0], [0, 0, 0]])
la13 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])
la23 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
ld1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
ld2 = (1/sqrt(3)) * np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]])

op_basis = [ls12, ls13, ls23, la12, la13, la23, ld1, ld2]

"""
# Dim 4 case
op_basis = []
for i in range(4):
    for j in range(4):
        if i == 0 and j == 0:
            continue
        else:
            op_basis.append(np.kron(sqp[i], sqp[j]))
  """          

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
        #state_ket_1 = qt.rand_ket_haar(2)
        #state_ket_2 = qt.rand_ket_haar(2)
        #state = np.kron(qt.ket2dm(state_ket_1).full(), qt.ket2dm(state_ket_2).full())

        freqs = mc_engine.simulate(bases, state, 10000)

        # Add to lbmle data set
        if with_lbmle:
            lbmle_freqs.append(freqs)

        # Flatten and add to NN data set
        flat_freqs = []
        for s in freqs:
            flat_freqs.extend(s)

        # Compute the basis coefficients; they should be real, but sometimes
        # there is a very very tiny (1e-17) complex part, so throw that away
    
        # For a Pauli basis
        #coefs = [np.trace(np.dot(x, state)).real for x in op_basis]

        # For the Gell-Mann basis in dimension 3
        coefs = [sqrt(3) * 0.5 * np.trace(np.dot(x, state)).real for x in op_basis]

        train_in.append(flat_freqs)
        train_out.append(coefs)

    return train_in, train_out, lbmle_freqs

    """# Split the data set into training and testing
    slice_point = int(n_trials * percent_train)
    test_in = np.array(train_in[0:slice_point])
    test_out = np.array(train_out[0:slice_point])
    train_in = np.array(train_in[slice_point:])
    train_out = np.array(train_out[slice_point:])

    if with_lbmle: 
        return train_in, train_out, test_in, test_out, lbmle_freqs[0:slice_point] 

    # No LBMLE data, just return training/testing data
    return train_in, train_out, test_in, test_out"""
    
