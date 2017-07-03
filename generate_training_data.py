from pynitefields import *
from balthasar import *

import qutip as qt
import numpy as np

from math import sqrt

def multiproc_generation(n_trials, dim, mc_engine, bases, op_basis):
    """ On a single processor, generate n_trials random states and frequencies and
        return it for processing by the master thread.
    """
    # Result storage
    input_frequencies = []
    output_coefs = [] 
    lbmle_frequencies = []

    for i in range(n_trials):
        state_ket = qt.rand_ket_haar(dim)
        state = qt.ket2dm(state_ket).full()

        freqs = mc_engine.simulate(bases, state)

        # Flatten to add to NN data set
        flat_freqs = []
        for s in freqs:
            flat_freqs.extend(s)

        coefs = []
        if dim % 2 == 0:
            coefs = [np.trace(np.dot(x, state)).real for x in op_basis]
        # For the Gell-Mann basis in dimension 3, traces are 2
        elif dim == 3:
            coefs = [sqrt(3) * 0.5 * np.trace(np.dot(x, state)).real for x in op_basis]
        else: 
            print("Invalid dimension in multiproc_generate.")

        # Update the data sets
        input_frequencies.append(flat_freqs)
        output_coefs.append(coefs)
        lbmle_frequencies.append(freqs)

    return input_frequencies, output_coefs, lbmle_frequencies


def generate_data(n_trials, n_workers, percent_test, f, op_basis, eigenvectors, bases):
    """ Generate training/testing data sets for our neural network, as well as store the
        testing data separately so we can compare it with the LBMLE reconstruction
        algorithm.

        n_trails - Total number of data points
        n_workers - Number of processes to spawn for generation (please make nicely divisible into n_trails)
        percent_test - How much of the data to return as testing data.
        f - The underlying finite field
        op_basis - The basis expansion in which we should compute the output coefficients
        eigenvectors - The MUBs in this dimension
        bases - The bases that we're measuring in.

        Returns train_in, train_out, test_in, test_out, and lbmle_freqs which is a copy of 
        test_in but suitable for use in the LBMLE reconstruction algorithm of Balthasar.
    """
    # Create the MUBs and MLE simulator
    mubs = MUBs(f) 
    mc_engine = LBMLE_MC(eigenvectors)

    # Set parameters for multiprocessing
    trials_per_worker = int(n_trials * 1./ n_workers) 
    input_params = [(trials_per_worker, f.dim, mc_engine, bases, op_basis)] * n_workers

    # Call in the troops!
    pool = Pool(processes = n_workers)
    all_data = pool.starmap(multiproc_generation, input_params)

    # Unfortunately what comes back is ugly. Thank god for stack overflow!
    # https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
    all_freqs_ugly = [x[0] for x in all_data]
    all_freqs = [x for y in all_freqs_ugly for x in y]
    
    all_coefs_ugly = [x[1] for x in all_data]
    all_coefs = [x for y in all_coefs for x in y]
         
    all_lbmle_freqs_ugly = [x[2] for x in all_data]
    all_lbmle_freqs = [x for y in all_lbmle_freqs_ugly for x in y]

    # Split the data set into training and testing
    slice_point = int(n_trials * percent_train)
    test_in = np.array(all_freqs[0:slice_point])
    test_out = np.array(all_coefs[0:slice_point])
    train_in = np.array(all_freqs[slice_point:])
    train_out = np.array(all_coefs[slice_point:])
    lbmle_freqs = np.array(all_lbmle_freqs[0:slice_point])

    return train_in, train_out, test_in, test_out, lbmle_freqs
