from pynitefields import *
from balthasar import *

import sys
import csv
import time

import numpy as np

# Import all the other things I wrote!
from eigvecs import *
from psd_utils import *
from gen_gell_mann_basis import *
from generate_training_data import *
from train_neural_net import *


def main():
    """ Run this script like the following:
        python nn_experiment.py dim n_trials n_workers percent_test bases filename
        
        Bases is going to be a list like 0,1,2,-1 etc. so that we can 
        parse it and send it to all the other functions.
    """
    if len(sys.argv) != 7:
        print("Not enough arguments provided")
        sys.exit()

    # Set up the finite field
    f = None
    eigenvectors = None

    d = int(sys.argv[1])

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
    else:
        print("Dimension not supported.")

    n_trials = int(sys.argv[2])
    n_workers = int(sys.argv[3])

    percent_test = float(sys.argv[4])

    bases = [int(x) for x in sys.argv[5].split(",")]
    op_basis = gen_gell_mann_basis(d)

    filename = sys.argv[6]
    
    print("Dimension: " + str(d))
    print("Num trials " + str(n_trials))
    print("Num workers " + str(n_workers))
    print("Percent test " + str(percent_test))
    print("Bases " + str(bases))

    hidden_layer_sizes = [128]

    results_nn = []
    actual_test_mats = []
    # Build the header for the output
    results = [["type"] + ["p" + str(i) for i in range(1, d**2)] + ["psd", "fidelity"]]

    t0 = time.time()

    # Generate all the data
    train_in, train_out, test_in, test_out, lbmle_freqs = generate_data(n_trials, n_workers, percent_test, 
                                                                        f, op_basis, eigenvectors, bases)

    print("Data generation complete.")
    print("Size of training set: " + str(len(train_in)))
    print("Size of testing set: " + str(len(test_in)) + "\n")

    for size in hidden_layer_sizes:
        print("Training neural network: ")
        my_nn = train_nn(train_in, train_out, size)

        # Normalize the predictions to the length of the Bloch ball in this dimension
        bloch_ball_pf = sqrt(1. * (d - 1) / (2 * d))
        predictions = my_nn.predict(test_in)
        scaled_predictions = [bloch_ball_pf * p / np.linalg.norm(p) for p in predictions]

        # Compute the fidelity with the test data
        fidelities_direct = []
        fidelities_psd = []


        for i in range(len(test_in)):
            # We are going to store data for three things:
            # The original test state, the state predicted by the NN, and the closest PSD state to the prediction
            next_results_test = ["test"]
            next_results_pred = ["pred"]
            next_results_closest = ["closest"]

            # Compute the density matrix using the reconstructed coefficients
            test_mat = (1./d)*np.eye(d) + np.sum([test_out[i][j] * op_basis[j] for j in range(d ** 2 - 1)], 0)
            pred_mat = (1./d)*np.eye(d) + np.sum([scaled_predictions[i][j] * op_basis[j] for j in range(d ** 2 - 1)], 0)

            # Now that we've computed the predicted matrix, compute the closest PSD matrix
            closest_psd = find_closest_psd(pred_mat)
            closest_coefs = [0.5 * np.trace(np.dot(x, closest_psd)).real for x in op_basis]

            # Add all the coefficients to their proper rows
            next_results_test.extend(test_out[i])
            next_results_pred.extend(scaled_predictions[i])
            next_results_closest.extend(closest_coefs)

            fidelities_direct.append(qt.fidelity(qt.Qobj(test_mat), qt.Qobj(pred_mat)))
            fidelities_psd.append(qt.fidelity(qt.Qobj(test_mat), qt.Qobj(closest_psd)))

            # Add information about whether or not the things are PSD
            if is_psd(test_mat):
                next_results_test.append(1)
            else:
                next_results_test.append(0)

            if is_psd(pred_mat):
                next_results_pred.append(1)
            else:
                next_results_pred.append(0)

            if is_psd(closest_psd): # Obviously this should be PSD, but still check
                next_results_closest.append(1)
            else:
                next_results_closest.append(0)

            # Add the fidelities
            next_results_test.append(1)
            next_results_pred.append(qt.fidelity(qt.Qobj(test_mat), qt.Qobj(pred_mat)))
            next_results_closest.append(qt.fidelity(qt.Qobj(test_mat), qt.Qobj(closest_psd)))

            # Update the results with this data point
            results.append(next_results_test)
            results.append(next_results_pred)
            results.append(next_results_closest)

            # Store the actual matrices to use in LBMLE section later
            actual_test_mats.append(test_mat) 

        # Finished going through all the test data!
        print("NN direct avg fidelity " + str(np.average(fidelities_direct)))
        print("NN PSD avg fidelity " + str(np.average(fidelities_psd)))

    # For each testing frequency, do LBMLE reconstruction and compute a fidelity
    do_mle = True
    results_lbmle = []
    if do_mle:
        mle = LBMLE(MUBs(f), eigenvectors)
   
        for i in range(len(lbmle_freqs)):
            next_results = ["lbmle"]

            mle_res = mle.estimate(bases, lbmle_freqs[i])

            # Compute the coefficients just to put in the table
            mle_coefs = [np.trace(np.dot(x, mle_res[0])).real for x in op_basis]

            fid = qt.fidelity(qt.Qobj(mle_res[0]), qt.Qobj(actual_test_mats[i]))

            next_results.extend(mle_coefs)

            if not is_psd(mle_res[0]):
                # This should *never* happen
                print("Oh dear, LBMLE reconstructed matrix is not PSD.")
                print("Matrix has eigenvalues ")
                print(np.linalg.eigvals(M))
                next_results.append(0)
            else:
                next_results.append(1)

            next_results.append(fid)
            results_lbmle.append(next_results)

        print("LBMLE avg fidelity = " + str(np.average([x[-1] for x in results_lbmle])))

    with open(filename, "w") as outfile:
        writer = csv.writer(outfile)
        for row in results: 
            writer.writerow(row)

        for row in results_lbmle:
            writer.writerow(row)

    for res in results_nn:
        print("Hidden layer size: " + str(res[0]) + " Avg fidelity = " + str(res[1]))

    t1 = time.time()

    print("Total execution time: " + str(t1 - t0))

if __name__ == '__main__':
    main()
