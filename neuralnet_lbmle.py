from pynitefields import *
from balthasar import *
from eigvecs_dim2 import eigenvectors

from scipy.linalg import sqrtm

from keras.layers import Input, Dense
from keras.models import Sequential

import keras.backend as K

from generate_training_data import *


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
N_TRIALS = 10000

f = GaloisField(2)
bases = [0, -1]
train_in, train_out, test_in, test_out, lbmle_freqs = generate_data(N_TRIALS, 0.1, f, eigenvectors, bases, True)

hidden_layer_sizes = [64]
results_nn = []
actual_test_mats = []
fidelities = []

for size in hidden_layer_sizes:
    print("Training neural network: ")
    my_nn = train_nn(train_in, train_out, size)

    predictions = my_nn.predict(test_in)
    scaled_predictions = [p / np.linalg.norm(p) for p in predictions]

    for i in range(len(test_in)):
        test_mat = (0.5*np.eye(2)) + 0.5*np.sum([test_out[i][j] * op_basis[j] for j in range(3)], 0)
        pred_mat = (0.5*np.eye(2)) + 0.5*np.sum([scaled_predictions[i][j] * op_basis[j] for j in range(3)], 0)

        actual_test_mats.append(test_mat) # Store the actual matrices to use in LBMLE section later

        fidelities.append(qt.fidelity(qt.Qobj(test_mat), qt.Qobj(pred_mat)))

        if i in range(25, 30):
            print("Actual matrix")
            print(test_mat)
            print(test_out[i])
            print("NN predicted matrix")
            print(pred_mat)
            print(scaled_predictions[i])
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
    
with open("fids_dim2.csv", "w") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["NN", "LBMLE"])
    for row in zip(fidelities, results_lbmle):
        writer.writerow(row)

for res in results_nn:
    print("Hidden layer size: " + str(res[0]) + " Avg fidelity = " + str(res[1]))

print("LBMLE avg fidelity = " + str(np.average(results_lbmle)))

