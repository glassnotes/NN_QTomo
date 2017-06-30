from pynitefields import *
from balthasar import *
from eigvecs_dim4 import eigenvectors

from scipy.linalg import sqrtm

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from keras.layers import Input, Dense, Dropout
from keras.models import Sequential

import keras.backend as K

from generate_training_data import *

from psd_finder import *

def is_pos_semidef(M):
    return np.all(np.linalg.eigvals(M) >= 0)


def train_nn(train_in, train_out, hidden_layer_size, dropout_pct = 0):
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

    # Add dropout if desired
    if dropout_pct != 0:
        model.add(Dropout(dropout_pct))
    
    # Add the output layer; need only one node that outputs a vector
    model.add(Dense(len(train_out[0]), activation = "tanh")) 
    
    # Cosine proximity gives us a measure of the angle between the true/predicted
    # values on the Bloch sphere.
    model.compile(optimizer = "sgd", loss = 'cosine_proximity')

    model.fit(train_in, train_out, epochs = 100, batch_size = 20, verbose = 2)

    return model 


# Actually create the neural network and do stuff.
N_TRIALS = 1000

f = GaloisField(2, 2, [1, 1, 1])
f.to_sdb([1, 2])

#bases = [1, 3, -1]
bases = [0, 1, 2, 3, -1]
train_in, train_out, test_in, test_out, lbmle_freqs = generate_data(N_TRIALS, 0.1, f, eigenvectors, bases, True)

pf = 1.0 / f.dim
num_params = f.dim ** 2 - 1 

hidden_layer_sizes = [8000]
dropout_pcts = [0.2, 0.3, 0.4, 0.5]
dropout_pcts = [0]
results_nn = []
actual_test_mats = []

fidelities_psd = []
fidelities_nonpsd = []


for pct in dropout_pcts:
    print("Training neural network: ")
    my_nn = train_nn(train_in, train_out, hidden_layer_sizes[0])

    predictions = my_nn.predict(test_in)
    scaled_predictions = [p / np.linalg.norm(p) for p in predictions]

    fidelities = []

    for i in range(len(test_in)):
        test_mat = (pf*np.eye(f.dim)) + pf*np.sum([test_out[i][j] * op_basis[j] for j in range(num_params)], 0)
        pred_mat = (pf*np.eye(f.dim)) + pf*np.sum([scaled_predictions[i][j] * op_basis[j] for j in range(num_params)], 0)

        actual_test_mats.append(test_mat) # Store the actual matrices to use in LBMLE section later

        # If the predicted matrix isn't positive semidefinite, find the closest thing that is
        if not is_pos_semidef(pred_mat):
            pred_mat = psd_finder(pred_mat)

        fid = qt.fidelity(qt.Qobj(test_mat), qt.Qobj(pred_mat))

        #if not is_pos_semidef(pred_mat):
        #    fidelities_nonpsd.append(fid)
        #jelse:
        #    fidelities_psd.append(fid)

        fidelities.append(fid)

        #if i in range(25, 30):
        #    print("Actual matrix")
        #    print(test_mat)
        #    print(test_out[i])
        #    print("NN predicted matrix")
        #    print(pred_mat)
        #    print(scaled_predictions[i])
            #print("Fidelity")
            #print(qt.fidelity(qt.Qobj(test_mat), qt.Qobj(pred_mat)))

    results_nn.append((pct, np.average(fidelities))) 

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
    
"""with open("fids_dim2.csv", "w") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["NN", "LBMLE"])
    for row in zip(fidelities, results_lbmle):
        writer.writerow(row)"""

for res in results_nn:
    print("Dropout percent: " + str(res[0]) + " Avg fidelity = " + str(res[1]))

#print("Mean fidelity PSD " + str(np.average(fidelities_psd)))
#print("Mean fidelity non-PSD " + str(np.average(fidelities_nonpsd)))

print("LBMLE avg fidelity = " + str(np.average(results_lbmle)))


regr_mor = MultiOutputRegressor(RandomForestRegressor(max_depth=30, random_state=0))
regr_mor.fit(train_in, train_out)

predictions = regr_mor.predict(test_in)
# Scale the predictions so that the vectors have norm one 
# i.e. make sure they are pure states on the surface of the Bloch sphere
results_mor = [p / np.linalg.norm(p) for p in predictions]
fidelities_mor = []
 
for i in range(len(test_in)):
    pred_mat = (pf*np.eye(f.dim)) + pf*np.sum([results_mor[i][j] * op_basis[j] for j in range(num_params)], 0)
    if not is_pos_semidef(pred_mat):
        pred_mat = psd_finder(pred_mat)
    fidelities_mor.append(qt.fidelity(qt.Qobj(actual_test_mats[i]), qt.Qobj(pred_mat)))

print("MOR avg fidelity = " + str(np.average(fidelities_mor)))

