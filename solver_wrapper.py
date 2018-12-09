from cvxopt import solvers, matrix
import numpy as np
from numpy import array as na

def solve_qp(*args):
    """
    wrap matlab quadprog solver by cvxopt.solvers.qp

    args: HQP(P), fQP(q), AQP(G), bQP(h), Aeq(A), beq(b), LB, UB

    minimize    (1/2)*x'*P*x + q'*x
    subject to  G*x > h
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

    P = matrix(args[0])
    q = matrix(args[1])
    # G = matrix(np.vstack([args[2], np.eye(n), -np.eye(n)]))
    # h = matrix(np.vstack([args[3], args[6], -args[7]]))
    G = matrix(np.vstack([args[2], np.eye(n)]))
    h = matrix(np.vstack([args[3], args[6]]))
    A = matrix(args[4])
    b = matrix(args[5])

    opts = {'kktreg':1e-10,
            'show_progress':False}


    # fake input
    # q = matrix(np.array([1.0,1.0,1.0]))

    # solved = solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b,
    #             kktsolver='ldl2', options=opts)

    solved = solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)

    return solved
