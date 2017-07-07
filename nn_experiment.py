from pynitefields import *
from balthasar import *

import sys
import csv
import time

from math import sqrt

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
    if len(sys.argv) != 5:
        print("Not enough arguments provided")
        sys.exit()

    data_in_file = sys.argv[1]
    data_out_file = sys.argv[2]
    percent_test = float(sys.argv[3])
    filename = sys.argv[4]

    # Grab the data from the files
    all_data_in = []
    all_data_out = []
    with open(data_in_file, "r") as infile:
        reader = csv.reader(infile)
        for row in reader:
            all_data_in.append([float(x) for x in row])
       
    with open(data_out_file, "r") as outfile:
        reader = csv.reader(outfile)
        for row in reader:
            all_data_out.append([float(x) for x in row])

    slice_point = int(percent_test * len(all_data_in))
    train_in = np.array(all_data_in[slice_point:])
    train_out = np.array(all_data_out[slice_point:])
    test_in = np.array(all_data_in[:slice_point])
    test_out = np.array(all_data_out[:slice_point])

    # Extract the system dimension from the data; should be sqrt(len + 1)
    d = int(sqrt(len(all_data_out[0]) + 1))

    op_basis = gen_gell_mann_basis(d)

    hidden_layer_sizes = [64, 1024, 2048, 4096]

    results_nn = []
    actual_test_mats = []

    # Build the header for the output
    results = [["type"] + ["p" + str(i) for i in range(1, d**2)] + ["psd", "fidelity"]]

    t0 = time.time()

    for size in hidden_layer_sizes:
        print("Training neural network: ")
        my_nn = train_nn(train_in, train_out, size)

        t1 = time.time()
        print("Neural network training time: " + str(t1 - t0))

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

        with open("hidden_" + str(size) + "_" + filename, "w") as outfile:
            writer = csv.writer(outfile)
            for row in results: 
                writer.writerow(row)

            #for row in results_lbmle:
            #    writer.writerow(row)

        for res in results_nn:
            print("Hidden layer size: " + str(res[0]) + " Avg fidelity = " + str(res[1]))
    """
    # For each testing frequency, do LBMLE reconstruction and compute a fidelity
    do_mle = False 
    results_lbmle = []
    t3 = time.time()
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
    """

    t4 = time.time()
    #print("LBMLE reconstruction time: " + str(t4 - t3))
    print("Total execution time: " + str(t4 - t0))

if __name__ == '__main__':
    main()
