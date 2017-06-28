from pynitefields import *
from balthasar import *
from eigvecs_dim3 import eigenvectors


from scipy.linalg import sqrtm

from keras.layers import Input, Dense
from keras.models import Sequential

import keras.backend as K

import numpy as np

from generate_training_data import *

def is_pos_semidef(M):
    return np.all(np.linalg.eigvals(M) >= 0)

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


# Actually create the neural network and do stuff.
N_TRIALS = 1000

f = GaloisField(3)
d = f.dim
bases = [0, 1, 2, -1]
train_in, train_out, test_in, test_out, lbmle_freqs = generate_data(N_TRIALS, 0.1, f, eigenvectors, bases, True)

hidden_layer_sizes = [64]
results_nn = []
actual_test_mats = []
angles = []

results = [["p1", "p2", "p3", "p4", "p5", "p6", "p7", "psd", "fidelity"]]

for size in hidden_layer_sizes:
    print("Training neural network: ")
    my_nn = train_nn(train_in, train_out, size)

    predictions = my_nn.predict(test_in)
    scaled_predictions = [p / np.linalg.norm(p) for p in predictions]

    fidelities = []


    for i in range(len(test_in)):
        next_results = []
        test_mat = ((1./d)*np.eye(d)) + (1./d)*np.sum([test_out[i][j] * op_basis[j] for j in range(d ** 2 - 1)], 0)
        pred_mat = ((1./d)*np.eye(d)) + (1./d)*np.sum([scaled_predictions[i][j] * op_basis[j] for j in range(d ** 2 - 1)], 0)

        actual_test_mats.append(test_mat) # Store the actual matrices to use in LBMLE section later

        angles.append(np.arccos(test_out[i][2]) / np.pi)
        fidelities.append(qt.fidelity(qt.Qobj(test_mat), qt.Qobj(pred_mat)))

        next_results.extend(scaled_predictions[i])

        # If the predicted matrix isn't positive semidefinite, find the closest thing that is
        if is_pos_semidef(pred_mat):
            next_results.append(1)
        else:
            next_results.append(0)

        next_results.append(qt.fidelity(qt.Qobj(test_mat), qt.Qobj(pred_mat)))
        results.append(next_results)
            
        

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

for i in range(len(lbmle_freqs)):
    mle_res = mle.estimate(bases, lbmle_freqs[i])
    fid = qt.fidelity(qt.Qobj(mle_res[0]), qt.Qobj(actual_test_mats[i]))
    results_lbmle.append(fid) 
    """
    if i == 25:
        print("Actual matrix")
        print(actual_test_mats[i])
        print("LBMLE predicted matrix: ")
        print(mle_res[0])
        print("Fidelity")
        print(fid)"""

with open("qutrit_output_1000.csv", "w") as outfile:
    writer = csv.writer(outfile)
    #writer.writerow(["Angle", "Fid_NN", "Fid_LBMLE"])
    #for row in zip(fidelities, results_lbmle):
    for row in results: 
        writer.writerow(row)

for res in results_nn:
    print("Hidden layer size: " + str(res[0]) + " Avg fidelity = " + str(res[1]))

print("LBMLE avg fidelity = " + str(np.average(results_lbmle)))

