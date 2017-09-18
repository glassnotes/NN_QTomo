import numpy as np
from math import sqrt

def extract_parameters(state):
    """ Turn the density matrix into something that looks like L^dag L
        with L lower diagonal and having d^2 - 1 parameters.
    """
    # First add the identity to make it positive definite
    d = state.shape[0]
    state = state + 10000 * np.eye(d)
    L = np.linalg.cholesky(state)

    ts = []
    for i in range(d):
        for j in range(i + 1): 
            ts.append(L[i][j].real)
            if i != j:
                ts.append(L[i][j].imag)

    return ts

def reconstruct_from_parameters(ts):
    """ Using a set of ts, reconstruct the original
        density matrix by constructing L, then L^dag L, 
        and finally subtracting the identity.
    """ d = int(sqrt(len(ts) + 1)) # Length of ts is d^2 - 1
    L = np.zeros((d, d), dtype = np.complex64)

    t_idx = 0
    for i in range(d):
        for j in range(i + 1):
            entry = ts[t_idx] + 0.j
            t_idx += 1
            
            if i != j:
                entry += 1j * ts[t_idx]
                t_idx += 1

            L[i][j] = entry
   
    state = np.dot(L, np.asmatrix(L).getH()) - 10000 * np.eye(d) 

    # Normalize the state before sending it back!
    return state / np.trace(state)


def proj(d, j, k): 
    """ Construct the matrix |j><k|.
        Helper for gen_gell_mann_basis.
     """
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
    
    return ggb 



def is_psd(M):
    """ Check if a matrix is positive semidefinite.
    """
    # Using only Hermitian matrices so use eigvalsh
    evs = np.linalg.eigvalsh(M)
  
    # Chop tiny numbers so -0.000000000000000003 doesn't make our matrix not PSD
    evs[np.abs(evs) < 1e-10] = 0
 
    return all(ev >= 0 for ev in evs)


def find_closest_psd(M):
    """ Input: M a Hermitian matrix.
        Output: The closest matrix to M (w.r.t. Euclidean/Frob. norm)
                that is positive semidefinite (PSD).

        We need such a function because an arbitrary generalized Bloch
        vector does not necessarily produce a legit quantum state;
        it is likely to not be PSD. So once our neural network is 
        done making predictions, use this to map the outputs to their
        closest PSD equivalents and see if this can boost the quality
        of the reconstruction.
    """

    # First we need to compute the eigensystem
    eigsys = np.linalg.eigh(M)
    eigvals, eigvecs = eigsys[0], eigsys[1]

    # Figure out the indices of the positive eigvalues
    pos_eigvals = [x for x in range(len(eigvals)) if eigvals[x] >= 0]

    PSDM = np.zeros_like(M)

    for x in pos_eigvals:
        next_eigvec = eigvecs[:,x]
        PSDM = PSDM + eigvals[x] * np.outer(next_eigvec, np.conj(next_eigvec))

    return PSDM 

def find_closest_pure(M):
    # First we need to compute the eigensystem
    eigsys = np.linalg.eigh(M)
    eigvals, eigvecs = eigsys[0], eigsys[1]

    # Figure out the indices of the positive eigvalues
    max_eigval = np.argmax(eigvals)


    max_eigvec = eigvecs[:,max_eigval]
    PSDM = eigvals[max_eigval] * np.outer(max_eigvec, np.conj(max_eigvec))
    PSDM = PSDM / np.trace(PSDM)

    return PSDM 
