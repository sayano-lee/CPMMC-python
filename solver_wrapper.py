from cvxopt import solvers, matrix
import numpy as np

def solve_qp(*args):
    """
    wrap matlab quadprog solver by cvxopt.solvers.qp

    args: HQP(P), fQP(q), AQP(G), bQP(h), Aeq(A), beq(b), LB, UB

    minimize    (1/2)*x'*P*x + q'*x
    subject to  G*x >= h
                A*x = b
                LB < x < UB

    dim: P  (n x n)
         q  (n x 1)
         G  (m x n)
         h  (m x 1)
         A  (p x n)
         b  (p x 1)
         LB (n x 1)
         UB (n x 1)

    """

    n = args[0].shape[0]

    P = args[0]
    q = args[1]
    G = np.vstack([args[2], -np.eye(n), np.eye(n)])
    h = np.vstack([args[3], args[6], args[7]])
    A = args[4]
    b = args[5]

    converted = map(matrix, [P, q, G, h, A, b])

    solved = solvers.qp(*converted)

    import ipdb
    ipdb.set_trace()

    return solved
