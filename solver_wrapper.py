from cvxopt import solvers, matrix
import numpy as np

def solve_qp(*args):
    """
    wrap matlab quadprog solver by cvxopt.solvers.qp

    :param args: HQP, fQP, AQP, bQP, Aeq, beq, LB, UB
    :return:
    """

    n = args[0].shape[0]

    P = args[0]
    q = args[1]
    import ipdb
    ipdb.set_trace()
    G = np.vstack([args[2], -np.eye(n), np.eye(n)])
    h = np.hstack([args[3], args[6], args[7]])
    A = args[4]
    b = args[5]

    converted = map(matrix, [P, q, G, h, A, b])

    solved = solvers.qp(*converted)

    import ipdb
    ipdb.set_trace()
