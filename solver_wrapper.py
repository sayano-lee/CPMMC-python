from cvxopt import solvers, matrix
import numpy as np

def solve_qp(*args):
    """
    wrap matlab quadprog solver by cvxopt.solvers.qp

    args: HQP, FQP, AQP, bQP, Aeq, beq, LB, UB

    minimize    (1/2)*x'*P*x + q'*x
    subject to  G*x >= h
                A*x = b
                LB < x < UB

    """

    n = args[0].shape[0]

    P = args[0]
    q = args[1]
    G = np.vstack([args[2], -np.eye(n), np.eye(n)])
    import ipdb
    ipdb.set_trace()
    h = np.hstack([args[3], args[6], args[7]])
    A = args[4]
    b = args[5]

    import ipdb
    ipdb.set_trace()

    converted = map(matrix, [P, q, G, h, A, b])

    solved = solvers.qp(*converted)

    import ipdb
    ipdb.set_trace()
