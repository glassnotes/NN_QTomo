from pynitefields import *
from balthasar import *
from eigvecs_dim2 import eigenvectors

from scipy.linalg import sqrtm

from keras.layers import Input, Dense
from keras.models import Sequential

from generate_training_data import *

def fidelity(X, Y):
   sqt = sqrtm(X)
   inner = np.dot(np.dot(sqt, Y), sqt)
   return np.trace(sqrtm(inner)).real


def train_nn(train_in, train_out):
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
    model.add(Dense(15, activation = "tanh", input_shape = i_shape))
    
    # Add the output layer; need only one node that outputs a vector
    model.add(Dense(len(train_out[0]), activation = "tanh")) 
    
    model.compile(optimizer = "sgd", metrics = ['accuracy'], loss = 'categorical_crossentropy')

    model.fit(train_in, train_out, epochs = 100, batch_size = 20, verbose = 2)

    return model 


# Actually create the neural network and do stuff.
N_TRIALS = 1000

f = GaloisField(2)
train_in, train_out, test_in, test_out = generate_data(N_TRIALS, 0.1, f, eigenvectors, [0, 1])

print("Training neural network: ")
my_nn = train_nn(train_in, train_out)

predictions = my_nn.predict(test_in)
scaled_predictions = [p / np.linalg.norm(p) for p in predictions]

fidelities = []
for i in range(len(test_in)):
    test_mat = (0.5*np.eye(2)) + 0.5*np.sum([test_out[i][j] * op_basis[j] for j in range(3)], 0)
    pred_mat = (0.5*np.eye(2)) + 0.5*np.sum([scaled_predictions[i][j] * op_basis[j] for j in range(3)], 0)
    
    fidelities.append(fidelity(test_mat, pred_mat))

    if i == 1:
        print(test_out[i])
        print(scaled_predictions[i])
   
print(fidelities[0:20])
print(np.average(fidelities))

