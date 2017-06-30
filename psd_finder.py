import numpy as np

def psd_finder(M):
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
