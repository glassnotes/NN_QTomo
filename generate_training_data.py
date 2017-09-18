import csv

import sys
import qutip as qt
import numpy as np

from multiprocessing import Pool

from mcexperiment import *
from eigvecs import *
from state_utils import *
from utils import *

def generate_projectors(eigenvectors):
    """ Converts a set of vectors into project form.
        Keeps order/format the same.
    """
    projectors = []
  
    for basis in eigenvectors:
        projectors_this_basis = []
        for vector in basis:
            projectors_this_basis.append(np.outer(vector, np.conj(vector)))
        projectors.append(projectors_this_basis)
    return projectors


def multiproc_generation(projs, n_trials, d, mc_engine, bases, op_basis):
    """ On a single processor, generate n_trials random states and frequencies and
        return it for processing by the master thread.
    """
    # Result storage
    input_frequencies = []
    output_coefs = [] 

    for i in range(n_trials):
        state_ket = qt.rand_ket_haar(d)
        state = qt.ket2dm(state_ket).full()

        #freqs = mc_engine.simulate(bases, state)
        freqs = [[float(np.real(np.trace(np.dot(state, projs[basis_idx][j])))) for j in range(len(projs[0]))] for basis_idx in range(len(bases))]

        # Flatten to add to NN data set
        flat_freqs = []
        for s in freqs:
            flat_freqs.extend(s)
        
        # Compute the Bloch vector coefficients using the Gell-Mann basis.
        # This will be a vector of length sqrt((d-1)/2d))
        # The factor of 0.5 is because the traces of the Gell-Mann ops are all 2
        #coefs = [0.5 * np.trace(np.dot(x, state)).real for x in op_basis]
        coefs = extract_parameters(state)

        # Update the data sets
        input_frequencies.append(flat_freqs)
        output_coefs.append(coefs)

    return input_frequencies, output_coefs 


def generate_data(n_trials, n_workers, d, op_basis, eigenvectors, bases):
    """ Generate training/testing data sets for our neural network, as well as store the
        testing data separately so we can compare it with the LBMLE reconstruction
        algorithm.

        n_trails - Total number of data points
        n_workers - Number of processes to spawn for generation (please make nicely divisible into n_trails)
        percent_test - How much of the data to return as testing data.
        op_basis - The basis expansion in which we should compute the output coefficients
        projs - The projectors of the MUBs in this dimension
        bases - The bases that we're measuring in.

        Returns train_in, train_out, test_in, test_out.
    """
    # Create the MUBs and MLE simulator
    mc_engine = MCExperiment(eigenvectors)

    projs = generate_projectors(eigenvectors)

    # Set parameters for multiprocessing
    trials_per_worker = int(n_trials * 1./ n_workers) 

    input_params = [(projs, trials_per_worker, d, mc_engine, bases, op_basis)] * n_workers

    # Call in the troops!
    pool = Pool(processes = n_workers)
    all_data = pool.starmap(multiproc_generation, input_params)

    # Unfortunately what comes back is ugly. Thank god for stack overflow!
    # https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
    all_freqs_ugly = [x[0] for x in all_data]
    all_freqs = [x for y in all_freqs_ugly for x in y]
    
    all_coefs_ugly = [x[1] for x in all_data]
    all_coefs = [x for y in all_coefs_ugly for x in y]
         
    return all_freqs, all_coefs 


def main():
    """ Collect user input on the amount of data to generate, and for which bases, etc.
        Then output this to input files and output files so we don't have to regenerate 
        hours worth of data every time we run the program.
    """
    if len(sys.argv) != 2:     
        print("Please run this script as ")
        print("python generatedata.py paramfile.txt")
        sys.exit()

    params = parse_param_file(sys.argv[1]) 
  
    # Set up the finite field
    eigenvectors = None

    d = params["DIM"]
  
    if d == 2:
        eigenvectors = eigenvectors_2
    elif d == 3:
        eigenvectors = eigenvectors_3
    elif d == 4:
        eigenvectors = eigenvectors_4
    elif d == 8:
        eigenvectors = eigenvectors_8
    elif d == 32:
        eigenvectors = eigenvectors_32
    else:
        print("Dimension not supported.")

    # Collect the bases
    bases = []
    if params["BASES"] == "all":
        bases = [x for x in range(d)] + [-1]
    else:
        bases = params["BASES"] 

    op_basis = gen_gell_mann_basis(d)

    train_in, train_out = generate_data(params["N_TRIALS"], params["N_WORKERS"], d, op_basis, eigenvectors, bases)

    with open(params["DATA_IN_FILE"], "w") as infile:
        writer = csv.writer(infile)
        writer.writerows(train_in)    

    with open(params["DATA_OUT_FILE"], "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(train_out)    

if __name__ == '__main__':
    main()
