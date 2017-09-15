from keras.layers import Input, Dense
from keras.models import Sequential

from math import sqrt
import numpy as np
from psd_utils import find_closest_psd
from gen_gell_mann_basis import *

class TomoNN():
    """ Build and train a neural network using the 'experimental data'.
        
        Input training data is a set of measurement frequencies.
        Output data is the expansion coefficients of the associated state
        in the Pauli basis.

        The network should in theory be able to deduce patterns in the
        values of the output coefficients using the probabilities of the
        input coefficients.
    """
    def __init__(self, d):
        """ Initialize the network to hold a Keras Sequential model. Only input variable
            we need is the dimension of the system.
        """
        self.d = d
        self.model = Sequential()


    def train(self, train_in, train_out, hidden_layer_size, percent_val = 0):
        """ Actually build and train the network using the provided training
            data and potentially the validation data.
        """
        # Define the shape of the input data, the vector of frequencies
        i_shape = train_in[0].shape
        inputs = Input(shape = i_shape)

        # Now begin building the model and add a single dense layer
        self.model.add(Dense(hidden_layer_size, activation = "tanh", input_shape = i_shape))
    
        # Add the output layer; need only one node that outputs a vector
        self.model.add(Dense(len(train_out[0]), activation = "tanh")) 
    
        # Cosine proximity gives us a measure of the angle between the true/predicted
        # values on the Bloch sphere.
        self.model.compile(optimizer = "sgd", loss = 'cosine_proximity')

        self.model.fit(train_in, train_out, epochs = 100, batch_size = 20, verbose = 2)


    def predict(self, test_in):
        """ Use the model we have built to make predictions based on the test data.
            
            This function will return a predicted density matrix for each piece of
            test data.           
 
            Making predictions involves not only passing things through the neural
            network to get the output vector, but also post-processing it so that 
            the outputs make physical sense, i.e. are positive semi-definite and    
            have trace 1.
        """
        op_basis = gen_gell_mann_basis(self.d)
        bloch_ball_pf = sqrt(1. * (self.d - 1) / (2 * self.d)) 
        base_predictions = self.model.predict(test_in)
         
        scaled_predictions = [bloch_ball_pf * p / np.linalg.norm(p) for p in base_predictions]

        pred_mats = [(1./self.d)*np.eye(self.d) + np.sum([p[j]*op_basis[j] for j in range(self.d**2-1)], 0) for p in scaled_predictions]
        
        closest_psds = [find_closest_psd(p) for p in pred_mats]
        closest_coefs = [[0.5 * np.trace(np.dot(x, p)).real for x in op_basis] for p in closest_psds]

        return closest_psds, closest_coefs
