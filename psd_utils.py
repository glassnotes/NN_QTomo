import numpy as np

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
