import numpy as np

from math import sqrt

def select_operator_basis(dim):
    """ Select the operator basis for expansion.
        Currently hard-coding in the possibilities for 3 dimensions:
            - 2: Single-qubit Paulis
            - 3: Gell-Mann matrices
            - 4: Two-qubit Paulis
        Function takes in a dimension and returns the corresponding 
        operator basis.
    """
    op_basis = []

    # Keep the Paulis here because they will be used for all 2^n cases.
    I = np.array([[1, 0], [0, 1]])
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    if dim == 2: # One qubit
        op_basis = [X, Y, Z]

    elif dim == 3: # One qutrit; Gell-Mann mats from Bertlmann and Kramer
        ls12 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        ls13 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
        ls23 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        la12 = np.array([[0,-1j, 0], [1j, 0, 0], [0, 0, 0]])
        la13 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])
        la23 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
        ld1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
        ld2 = (1/sqrt(3)) * np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]])
    
        op_basis = [ls12, ls13, ls23, la12, la13, la23, ld1, ld2]

    elif dim == 4: # Two qubits / one ququart?
        sqp = [I, X, Y, Z] # Collect single-qubit Paulis
        for i in range(4): # Take tensor products of all of them
            for j in range(4):
                if i == 0 and j == 0:
                    continue
                else:
                    op_basis.append(np.kron(sqp[i], sqp[j]))
    else:
        print("Higher-dimensional systems currently not implemented.")

    return op_basis


