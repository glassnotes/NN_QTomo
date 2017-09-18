import os, sys
import csv
import time 
from math import sqrt

import numpy as np

# Import all the other things I wrote!
from quinton import *

def main():
    """Runs a neural network configured as per input param file.
    """ 
    if len(sys.argv) != 2:
        print("Please run this script as ")
        print("python nn_experiment.py paramfile.txt")
        sys.exit()

    params = parse_param_file(sys.argv[1])

    # Grab the data from the files
    all_data_in = []
    all_data_out = []

    # If the corresponding data files don't already exist,
    # we will need to generate them.
    if not os.path.exists('./' + params["DATA_IN_FILE"]):
        print("Data file not found; generating data.")
        dg = TomoDataGenerator(params)
        dg.generate()
        
    # Now read in the data.
    with open(params["DATA_IN_FILE"], "r") as infile:
        reader = csv.reader(infile)
        for row in reader:
            all_data_in.append([float(x) for x in row])
       
    with open(params["DATA_OUT_FILE"], "r") as outfile:
        reader = csv.reader(outfile)
        for row in reader:
            all_data_out.append([float(x) for x in row])

    # Split into training, testing, and validation
    test_slice_point = int(params["PERCENT_TEST"] * params["N_TRIALS"])
    val_slice_point = int(test_slice_point + params["PERCENT_VAL"] * params["N_TRIALS"])

    test_in = np.array(all_data_in[:test_slice_point])
    test_out = np.array(all_data_out[:test_slice_point])
    val_in = np.array(all_data_in[test_slice_point:val_slice_point])
    val_out = np.array(all_data_in[test_slice_point:val_slice_point])
    train_in = np.array(all_data_in[val_slice_point:])
    train_out = np.array(all_data_out[val_slice_point:])

    # Pipe standard output to the log file.
    sys.stdout = open(params["LOG_FILE"], 'w')

    print("Data loading complete.")
    print("Network and data parameters: ")
    print("DIMENSION: " + str(params["DIM"]))
    print("BASES: " + str(params["BASES"]))
    
    train_size = params["N_TRIALS"] - params["N_TRIALS"]*(params["PERCENT_TEST"] + params["PERCENT_VAL"])
    print("TRAIN_SIZE:" + str(train_size)) 
    print("VAL_SIZE: " + str(params["PERCENT_VAL"] * params["N_TRIALS"]))
    print("TEST_SIZE: " + str(params["PERCENT_TEST"] * params["N_TRIALS"]))

    # Extract the system dimension from the data; should be sqrt(len + 1)
    d = params["DIM"]
    if d != int(sqrt(len(all_data_out[0]) + 1)):
        print("Error, mismatch between dimension parameter and output data.")
    
    results_nn = []
    actual_test_mats = []

    # Build the header for the output
    results = [["type"] + ["p" + str(i) for i in range(1, d**2)] + ["psd", "fidelity"]]

    for size in params["HIDDEN_LAYER_SIZES"]:
        print("Training neural network: ")
        t0 = time.time()

        my_nn = TomoNeuralNetwork(params) 
        my_nn.train(size, train_in, train_out)

        t1 = time.time()
        print("Hidden layer size: " + str(size))
        print("Neural network training time: " + str(t1 - t0))

        # Create the output states for all of the test data.
        test_psds = [(1./d)*np.eye(d) + np.sum([t[j] * params["OP_BASIS"][j] for j in range(d ** 2 - 1)], 0) for t in test_out]

        closest_psds, closest_coefs = my_nn.predict(test_in)

        # Compute the fidelity with the test data
        fidelities_psd = [qt.fidelity(qt.Qobj(test_psds[i]), qt.Qobj(closest_psds[i])) for i in range(len(test_psds))]

        print(closest_psds[0])
        print(closest_coefs[0])
        print(test_psds[0])
        print(test_out[0])

        # Finished going through all the test data!
        print("NN PSD avg fidelity " + str(np.average(fidelities_psd)))

        outfile_name = params["LOG_FILE"][:-4] + "_h" + str(size) + ".pred"
        with open(outfile_name, "w") as outfile:
            writer = csv.writer(outfile)
            for row in results: 
                writer.writerow(row)


if __name__ == '__main__':
    main()
