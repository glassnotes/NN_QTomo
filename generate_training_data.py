from pynitefields import *
from balthasar import *

import csv

import qutip as qt
import numpy as np

from multiprocessing import Pool

from eigvecs import *
from gen_gell_mann_basis import *
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


def multiproc_generation(eigenvectors, n_trials, d, mc_engine, bases, op_basis):
    """ On a single processor, generate n_trials random states and frequencies and
        return it for processing by the master thread.
    """
    # Result storage
    input_frequencies = []
    output_coefs = [] 
    lbmle_frequencies = []

    projs = generate_projectors(eigenvectors) 


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
        coefs = [0.5 * np.trace(np.dot(x, state)).real for x in op_basis]

        # Update the data sets
        input_frequencies.append(flat_freqs)
        output_coefs.append(coefs)
        lbmle_frequencies.append(freqs)

    return input_frequencies, output_coefs, lbmle_frequencies


def generate_data(n_trials, n_workers, f, op_basis, eigenvectors, bases):
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
    input_params = [(eigenvectors, trials_per_worker, f.dim, mc_engine, bases, op_basis)] * n_workers

    # Call in the troops!
    pool = Pool(processes = n_workers)
    all_data = pool.starmap(multiproc_generation, input_params)

    # Unfortunately what comes back is ugly. Thank god for stack overflow!
    # https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
    all_freqs_ugly = [x[0] for x in all_data]
    all_freqs = [x for y in all_freqs_ugly for x in y]
    
    all_coefs_ugly = [x[1] for x in all_data]
    all_coefs = [x for y in all_coefs_ugly for x in y]
         
    all_lbmle_freqs_ugly = [x[2] for x in all_data]
    all_lbmle_freqs = [x for y in all_lbmle_freqs_ugly for x in y]

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
    f = None
    eigenvectors = None

    d = params["DIM"]
  
    if d == 2:
        f = GaloisField(d)
        eigenvectors = eigenvectors_2
    elif d == 3:
        f = GaloisField(d)
        eigenvectors = eigenvectors_3
    elif d == 4:
        f = GaloisField(2, 2, [1, 1, 1])
        f.to_sdb([1, 2])
        eigenvectors = eigenvectors_4
    elif d == 8:
        f = GaloisField(2, 3, [1, 1, 0, 1])
        f.to_sdb([3, 5, 6])
        eigenvectors = eigenvectors_8
    elif d == 32:
        f = GaloisField(2, 5, [1, 0, 1, 0, 0, 1])
        f.to_sdb([3, 5, 11, 22, 24])
        from eigvecs_32 import eigenvectors_32
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

    train_in, train_out = generate_data(params["N_TRIALS"], params["N_WORKERS"], f, op_basis, eigenvectors, bases)

    with open(params["DATA_IN_FILE"], "w") as infile:
        writer = csv.writer(infile)
        writer.writerows(train_in)    

    with open(params["DATA_OUT_FILE"], "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(train_out)    

if __name__ == '__main__':
    main()
