import numpy as np
from itertools import product

# TODO document

def similarity_loss(matrix, order):
    rcs = np.arange(matrix.shape[0])
    ordered = matrix[order]
    return sum((dot(ordered[i],ordered[j]))*((i-j)**2) for i,j in product(rcs,rcs))

def schedule(t, alpha=.99, T0=1e7):
    return T0*(alpha**t)

def successor(order):
    rcs = np.random.permutation(order.shape[0])[:2]
    suc = order.copy()
    suc[rcs[0]] = order[rcs[1]]
    suc[rcs[1]] = order[rcs[0]]
    return suc

def order_rows_by_sim(matrix, tol=1e-7, loss = similarity_loss):
    order = np.arange(matrix.shape[0])
    E = loss(matrix, order)
    for t in count():
        # Cool things down
        T = schedule(t)
        if T < tol: break
        # Look ahead
        succ = successor(order)
        Esuc = loss(matrix, succ)
        deltaE = Esuc - E
        # Accept or reject based on energy
        if np.random.rand() < min(1, np.exp(-deltaE/T)):
            order = succ
            E = Esuc
    return order
