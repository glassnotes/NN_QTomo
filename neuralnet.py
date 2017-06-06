from pynitefields import *
from balthasar import *

from keras.layers import Input, Dense
from keras.models import Sequential

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
    model.add(Dense(15, activation = "linear", input_shape = i_shape))
    
    # Add the output layer; need only one node that outputs a vector
    model.add(Dense(len(train_out[0]), activation = "linear")) 
    
    model.compile(optimizer = "sgd", metrics = ['accuracy'], loss = 'categorical_crossentropy')

    fit = model.fit(train_in, train_out, epochs = 100, batch_size = 20, verbose = 2)

    return fit

