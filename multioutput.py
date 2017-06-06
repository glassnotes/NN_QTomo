import numpy as numpy

from eigvecs_dim2 import eigenvectors
from pynitefields import *

from scipy.linalg import sqrtm

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from generate_training_data import *

def fidelity(X, Y):
    inner = np.dot(np.dot(sqrtm(X), Y), sqrtm(X))
    return np.trace(sqrtm(inner)).real

N_TRIALS = 10000

f = GaloisField(2)

train_in, train_out, test_in, test_out = generate_data(N_TRIALS, 0.01, f, eigenvectors, [0, 1])

regr_mor = MultiOutputRegressor(RandomForestRegressor(max_depth=30, random_state=0))
regr_mor.fit(train_in, train_out)

predictions = regr_mor.predict(test_in)

# Scale the predictions so that the vectors have norm one 
# i.e. make sure they are pure states on the surface of the Bloch sphere
scaled_predictions = [p / np.linalg.norm(p) for p in predictions]

fidelities = []

for i in range(len(test_in)):
    test_mat = (0.5*np.eye(2)) + 0.5*np.sum([test_out[i][j] * op_basis[j] for j in range(3)], 0)
    pred_mat = (0.5*np.eye(2)) + 0.5*np.sum([scaled_predictions[i][j] * op_basis[j] for j in range(3)], 0)

    #print(test_mat)
    #print(pred_mat)
    fidelities.append(fidelity(pred_mat, test_mat))

print(fidelities[0:20])
print(np.average(fidelities))
