from pynitefields import *
from balthasar import *
from eigvecs_dim3 import eigenvectors
from psd_finder import *

from scipy.linalg import sqrtm

from keras.layers import Input, Dense
from keras.models import Sequential

import keras.backend as K

import numpy as np

from generate_training_data import *

def is_pos_semidef(M):
    # Using only Hermitian matrices so use eigvalsh
    evs = np.linalg.eigvalsh(M)

    # Chop tiny numbers so -0.000000000000000003 doesn't make our matrix not PSD
    evs[np.abs(evs) < 1e-10] = 0
    
    return all(ev >= 0 for ev in evs)

def train_nn(train_in, train_out, hidden_layer_size):
    """ Build and train a neural network using the 'experimental data'.
        
        Input training data is a set of measurement frequencies.
        Output data is the expansion coefficients of the associated state
        in the Pauli basis.

        The network should in theory be able to deduce patterns in the
        values of the output coefficients using the probabilities of the
        input coefficients.
    """

    # Define the shape of the input data, the vector of frequencies
    i_shape = train_in[0].shape
    inputs = Input(shape = i_shape)

    # Now begin building the model and add a single dense layer
    model = Sequential()
    model.add(Dense(hidden_layer_size, activation = "tanh", input_shape = i_shape))
    
    # Add the output layer; need only one node that outputs a vector
    model.add(Dense(len(train_out[0]), activation = "tanh")) 
    
    # Cosine proximity gives us a measure of the angle between the true/predicted
    # values on the Bloch sphere.
    model.compile(optimizer = "sgd", loss = 'cosine_proximity')

    model.fit(train_in, train_out, epochs = 100, batch_size = 20, verbose = 2)

    return model 

# Actually create the neural network and do stuff
N_TRIALS = 1000

f = GaloisField(3)
d = f.dim
bases = [0, 1, 2, -1]


# Use parallel processing to generate the training data
from multiprocessing import Pool
num_workers = 10
percent_train = 0.1

input_params = [(int(N_TRIALS / num_workers), percent_train, f, eigenvectors, bases, True)] * num_workers

import time
t0 = time.time()

pool = Pool(processes = num_workers)
all_data = pool.starmap(generate_data, input_params)

# Thank god for stack overflow
# https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
all_freqs_ugly = [x[0] for x in all_data]
all_freqs = [x for y in all_freqs_ugly for x in y]

all_vectors_ugly = [x[1] for x in all_data]
all_vectors = [x for y in all_vectors_ugly for x in y]

all_lbmle_freqs_ugly = [x[2] for x in all_data]
all_lbmle_freqs = [x for y in all_lbmle_freqs_ugly for x in y]

#train_in, train_out, test_in, test_out, lbmle_freqs = generate_data(N_TRIALS, 0.1, f, eigenvectors, bases, True)

# Split the data set into training and testing
slice_point = int(N_TRIALS * percent_train)
test_in = np.array(all_freqs[0:slice_point])
test_out = np.array(all_vectors[0:slice_point])
train_in = np.array(all_freqs[slice_point:])
train_out = np.array(all_vectors[slice_point:])
lbmle_freqs = np.array(all_lbmle_freqs[0:slice_point]) 

hidden_layer_sizes = [64]
results_nn = []
actual_test_mats = []
angles = []

results = [["type", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "original_norm", "psd", "fidelity"]]

for size in hidden_layer_sizes:
    print("Training neural network: ")
    my_nn = train_nn(train_in, train_out, size)

    predictions = my_nn.predict(test_in)
    scaled_predictions = [p / np.linalg.norm(p) for p in predictions]

    fidelities = []

    for i in range(len(test_in)):
        next_results_test = ["test"]
        next_results_pred = ["pred"]
        # For the Pauli basis, at least, for one qubit this works fine
        #test_mat = ((1./d)*np.eye(d)) + (1./d)*np.sum([test_out[i][j] * op_basis[j] for j in range(d ** 2 - 1)], 0)
        #pred_mat = ((1./d)*np.eye(d)) + (1./d)*np.sum([scaled_predictions[i][j] * op_basis[j] for j in range(d ** 2 - 1)], 0)

        # For the qutrit Gell-Mann basis
        test_mat = ((1./d)*np.eye(d)) + (1./sqrt(d))*np.sum([test_out[i][j] * op_basis[j] for j in range(d ** 2 - 1)], 0)
        pred_mat = ((1./d)*np.eye(d)) + (1./sqrt(d))*np.sum([scaled_predictions[i][j] * op_basis[j] for j in range(d ** 2 - 1)], 0)

        actual_test_mats.append(test_mat) # Store the actual matrices to use in LBMLE section later

        angles.append(np.arccos(test_out[i][2]) / np.pi)
        fidelities.append(qt.fidelity(qt.Qobj(test_mat), qt.Qobj(pred_mat)))

        next_results_test.extend(test_out[i])
        next_results_pred.extend(scaled_predictions[i])

        # Add the original norm so we can see how much it got scaled by
        print(test_out[i])
        next_results_test.append(np.linalg.norm(test_out[i]))
        next_results_pred.append(np.linalg.norm(predictions[i]))

        # If the predicted matrix isn't positive semidefinite, find the closest thing that is
        if is_pos_semidef(test_mat):
            next_results_test.append(1)
        else:
            next_results_test.append(0)

        # If the predicted matrix isn't positive semidefinite, find the closest thing that is
        if is_pos_semidef(pred_mat):
            next_results_pred.append(1)
        else:
            next_results_pred.append(0)

        next_results_test.append(1)
        next_results_pred.append(qt.fidelity(qt.Qobj(test_mat), qt.Qobj(pred_mat)))

        results.append(next_results_test)
        results.append(next_results_pred)

        # Find the closest PSD matrix and also compare
        next_results_closest = ["closest"]
        closest_psd = psd_finder(pred_mat)
        closest_coefs = [sqrt(3) * 0.5 * np.trace(np.dot(x, closest_psd)).real for x in op_basis]
        next_results_closest.extend(closest_coefs)
        next_results_closest.append(np.linalg.norm(closest_coefs))
        if is_pos_semidef(closest_psd):
            next_results_closest.append(1)
        else:
            next_results_closest.append(0)
        next_results_closest.append(qt.fidelity(qt.Qobj(test_mat), qt.Qobj(closest_psd)))

        results.append(next_results_closest)

        
            
        

        #if i in range(25, 30):
        #    print("Actual matrix")
        #    print(test_mat)
        #    print(test_out[i])
        #    print("NN predicted matrix")
        #    print(pred_mat)
        #    print(scaled_predictions[i])
            #print("Fidelity")
            #print(qt.fidelity(qt.Qobj(test_mat), qt.Qobj(pred_mat)))

    results_nn.append((size, np.average(fidelities))) 

# For each testing frequency, do LBMLE reconstruction and compute a fidelity
results_lbmle = []
mle = LBMLE(MUBs(f), eigenvectors)
"""
for i in range(len(lbmle_freqs)):
    mle_res = mle.estimate(bases, lbmle_freqs[i])
    if not is_pos_semidef(mle_res[0]):
        print("Oh dear, LBMLE reconstructed matrix is not PSD.")
        print("Matrix has eigenvalues ")
        print(np.linalg.eigvals(M))
    fid = qt.fidelity(qt.Qobj(mle_res[0]), qt.Qobj(actual_test_mats[i]))
    results_lbmle.append(fid) 
   
    if i == 25:
        print("Actual matrix")
        print(actual_test_mats[i])
        print("LBMLE predicted matrix: ")
        print(mle_res[0])
        print("Fidelity")
        print(fid)
"""

with open("qutrit_output_100000_with_closest.csv", "w") as outfile:
    writer = csv.writer(outfile)
    #writer.writerow(["Angle", "Fid_NN", "Fid_LBMLE"])
    #for row in zip(fidelities, results_lbmle):
    for row in results: 
        writer.writerow(row)

for res in results_nn:
    print("Hidden layer size: " + str(res[0]) + " Avg fidelity = " + str(res[1]))

#print("LBMLE avg fidelity = " + str(np.average(results_lbmle)))

t1 = time.time()

print("Total execution time: " + str(t1 - t0))
