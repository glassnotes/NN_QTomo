from pynitefields import *
from balthasar import *

import qutip 

import csv

from eigvecs_dim4 import eigenvectors

import numpy as np

from pprint import pprint
np.set_printoptions(precision=4, suppress=True)

from scipy.linalg import sqrtm

N_TRIALS = 1000

f = GaloisField(2, 2, [1, 1, 1])
f.to_sdb([1, 2])

mubs = MUBs(f) 

mle = LBMLE(mubs, eigenvectors)

def fidelity(X, Y):
    inner = np.dot(np.dot(sqrtm(X), Y), sqrtm(X))
    return np.trace(sqrtm(inner))

bases1 = [0, 3, -1] # One version of coarse-graining
bases2 = [1, 2, -1] # The other two bases, plus one of the random ones 
bases3 = [0, 3]
bases4 = [1, 2]


n_1 = ["conv01m1"]
n_2 = ["conv23m1"]
n_3 = ["conv01"]
n_4 = ["conv23"]
fid_1 = ["fid01m1"]
fid_2 = ["fid23m1"]
fid_3 = ["fid01"]
fid_4 = ["fid23"]

for i in range(N_TRIALS):
    if i % 1000 == 0:
        print(i)
    # Generate a random state with qutip
    state = qutip.rand_dm(4).full()

    #print("Initial random state: ")
    #pprint(state)

    mc_engine = LBMLE_MC(eigenvectors)

    freqs1 = mc_engine.simulate(bases1, state)
    freqs2 = mc_engine.simulate(bases2, state)
    freqs3 = mc_engine.simulate(bases3, state)
    freqs4 = mc_engine.simulate(bases4, state)

    # Do the estimates 
    rho1, n1 = mle.estimate(bases1, freqs1)
    rho2, n2 = mle.estimate(bases2, freqs2)
    rho3, n3 = mle.estimate(bases3, freqs3)
    rho4, n4 = mle.estimate(bases4, freqs4)

    """print("Estimate 1: ")
    pprint(rho1)
    print("Estimate 2: ")
    pprint(rho2)
    print("Estimate 3: ")
    pprint(rho3)"""

    n_1.append(n1)
    n_2.append(n2)
    n_3.append(n3)
    n_4.append(n4)
    fid_1.append(fidelity(rho1, state).real)
    fid_2.append(fidelity(rho2, state).real)
    fid_3.append(fidelity(rho3, state).real)
    fid_4.append(fidelity(rho4, state).real)

ids = ["ID"] + [x for x in range(N_TRIALS)]

rows = zip(ids, n_1, n_2, n_3, n_4, fid_1, fid_2, fid_3, fid_4) 

with open("dim4_prelim.csv", 'w') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)

#print("Average number of steps for bases 1: " + str(np.mean(n_1)))
#print("Average number of steps for bases 2: " + str(np.mean(n_2)))
#print("Average number of steps for bases 3: " + str(np.mean(n_3)))
#print("Average fidelity for bases 1: " + str(np.mean(fid_1)))
#print("Average fidelity for bases 2: " + str(np.mean(fid_2)))
#print("Average fidelity for bases 3: " + str(np.mean(fid_3)))
