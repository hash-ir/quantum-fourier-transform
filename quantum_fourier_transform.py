#!/usr/bin/env python
# coding: utf-8

import numpy as np
np.set_printoptions(suppress=True) # print without scientific notation

# define single qubit computational basis states
q0 = [1, 0] # |0>
q1 = [0, 1] # |1>

# hadamard gate
H = np.array([[1, 1],
              [1, -1]]) / np.sqrt(2)

# 1 qbit identity gate
I = np.eye(2)

# phase gate S
S = np.array([[1,0],
              [0, 1j]])

# controlled-S gate
controlled_S = np.block([[I, np.zeros((2, 2))],
                       [np.zeros((2,2)), S]])

# pi/8 gate T
T = np.array([[1,0],
              [0, np.exp(1j*np.pi/4)]])

# controlled-T gate
controlled_T = np.block([[I, np.zeros((2, 2))],
                        [np.zeros((2, 2)), T]])

# 2 qbit swap gate
swap = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]])

def reorder_gate(G, perm):
    """
    Adapt gate G to an ordering of the 
    qubits as specified in perm.
    Example, given G = np.kron(np.kron(A, B), C):
    reorder_gate(G, [1, 2, 0]) == np.kron(np.kron(B, C), A)
    """
    perm = list(perm)
    # number of qubits
    n = len(perm)
    # reorder both input and output dimensions
    perm2 = perm + [n + i for i in perm]
    return np.reshape(np.transpose(np.reshape(G, 2*n*[2]), perm2), (2**n, 2**n))

def make_multiple_qbits(qbits=[0,0,0]):
    """
    Performs tensor product on individual 
    qbits to represent n-qubit state
    
    Args:
        qbits (np.array/list): 3 qbit state in ket notation, shape [3]
        
    Returns:
        result (np.array/list): 3 qbit state vector, shape [8]
    """
    # intialize with first qbit
    if qbits[0] == 0:
        result = q0
    else:
        result = q1
        
    # tensor product with remaining qbits
    for q in qbits[1:]:
        if q == 0:
            result = np.kron(result, q0)
        else:
            result = np.kron(result, q1)
        
    return result


def fourier_transform():
    """
    Construct the operators and compute 
    the fourier transform matrix
        
    Returns:
        QFT (np.array): matrix multiplication of operators, shape [8, 8]
    """
    t1 = np.kron(np.kron(H, I), I) # hadamard on the first qbit
    
    t2 = np.kron(controlled_S, I)
    t2 = reorder_gate(t2, [1,0,2]) # control bit=2, target bit=1 
    
    t3 = np.kron(controlled_T, I)
    t3 = reorder_gate(t3, [1,2,0]) # control bit=3, target bit=1
    
    t4 = np.kron(np.kron(I, H), I) # hadamard on the second qbit
    
    t5 = np.kron(controlled_S, I)
    t5 = reorder_gate(t5, [2,1,0]) # control bit=3, target bit=2
    
    t6 = np.kron(np.kron(I, I), H) # hadamard on the third qbit
    
    t7 = np.kron(swap, I)
    t7 = reorder_gate(t7, [0,2,1]) # swap qbits 1 and 3
    
    QFT = t1
    
    for op in (t2, t3, t4, t5, t6, t7):
        QFT = np.matmul(QFT, op) # matrix multiplication of operators
    
    return QFT

QFT_ref = np.array([[np.exp(2*np.pi*1j*j*k/8)/np.sqrt(8) for j in range(8)] for k in range(8)])
QFT = fourier_transform()
print("Reference:", QFT_ref)
print()
print("Computed:", QFT)

qbits = [0,0,1]
psi = make_multiple_qbits(qbits)
psi1 = QFT_ref.dot(psi)
psi2 = QFT.dot(psi)

print("Initial state:", psi)
print()
print("Transformed state (reference):", psi1)
print()
print("Transformed state (computed):", psi2)




