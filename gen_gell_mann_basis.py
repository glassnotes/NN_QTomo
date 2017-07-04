import numpy as np

from math import sqrt
from pprint import pprint

def proj(d, j, k):
    """ Construct the matrix |j><k| """
    M = np.zeros((d, d)) 
    M[j, k] = 1;
    return M

def gen_gell_mann_basis(d):
    """ Generate the generalized Gell-Mann matrices in dimension d.
        These are used
    """
    ggb = []

    # Generate the symmetric matrices
    #       |j><k| + |k><j|
    for j in range(d):
        for k in range(j + 1, d):
            ggb.append(proj(d, j, k) + proj(d, k, j))

    # Generate the antisymmetric matrices
    #      -i |j><k| + i |k><j|
    for j in range(d):
        for k in range(j + 1, d):
            ggb.append(-1j * proj(d, j, k) + 1j * proj(d, k, j))

    # Finally the diagonal matries
    # sqrt(2/l(l+1))( sum_j=1^l |j><j| - l |l+1><l+1| )
    for l in range(1, d):
        prefactor = sqrt(2 / (l * (l + 1)))

        sum_over_j = np.zeros((d, d))
        for j in range(l):
            sum_over_j = sum_over_j + proj(d, j, j)
    
        ggb.append(prefactor * (sum_over_j - l * proj(d, l, l)))
        
    for op in ggb:
        pprint(op)
        
    return ggb 


