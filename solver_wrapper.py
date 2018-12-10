from cvxopt import solvers, matrix
import numpy as np


def check_inf_sanity_in_np_array(arr):
    flag = len([i for i in range(len(arr)) if (arr[i].item() == float('inf'))])

    if flag == 0:
        return 0
    if flag == len(arr):
        return 1

    return -1


def solve_qp(P, q, G=None, h=None, A=None, b=None,
             LB=None, UB=None, kktsolver=None):
    """
    wrap matlab quadprog solver by cvxopt.solvers.qp

    args: HQP(P), fQP(q), AQP(G), bQP(h), Aeq(A), beq(b), LB, UB
          kktsolver, opts

    minimize    (1/2)*x'*P*x + q'*x
    subject to  G*x <= h
                A*x = b
                LB <= x <= UB

    dim: P  (n x n)
         q  (n x 1)
         G  (m x n)
         h  (m x 1)
         A  (p x n)
         b  (p x 1)
         LB (n x 1)
         UB (n x 1)

    """

    ## TODO: pass opts as a variable not hard-coded one
    opts = {"show_progress": False}
    n = P.shape[0]
    P = matrix(P)
    q = matrix(q)

    A = matrix(A) if A is not None else None
    b = matrix(b) if b is not None else None

    # Wrap G/h with LB/UB
    if LB is not None and UB is not None:

        # check inf sanity, if LB/UP -inf or inf, remove constraints
        flag_LB = check_inf_sanity_in_np_array(LB)
        flag_UB = check_inf_sanity_in_np_array(UB)

        # check if G/h exists and wrap LB/UB by G/h
        if G is not None and h is not None:
            if flag_LB == 0 and flag_UB == 0:
                G = matrix(np.vstack([G, -np.eye(n), np.eye(n)]))
                h = matrix(np.vstack([h, -LB, UB]))
            if flag_LB == -1 and flag_UB != -1:
                raise ValueError("partial LB inf not supported")
            if flag_UB == -1 and flag_LB != -1:
                raise ValueError("partial UB inf not supported")
            if flag_LB == -1 and flag_UB == -1:
                raise ValueError("partial LB and UB inf not supported")
            if flag_LB == 1 and flag_UB != -1:
                G = matrix(np.vstack([G, np.eye(n)]))
                h = matrix(np.vstack([h, UB]))
            if flag_UB == 1 and flag_LB != -1:
                G = matrix(np.vstack([G, -np.eye(n)]))
                h = matrix(np.vstack([h, -LB]))
        if G is None and h is None:
            if flag_LB == 0 and flag_UB == 0:
                G = matrix(np.vstack([-np.eye(n), np.eye(n)]))
                h = matrix(np.vstack([-LB, UB]))
            if flag_LB == -1 and flag_UB != -1:
                raise ValueError("partial LB inf not supported")
            if flag_UB == -1 and flag_LB != -1:
                raise ValueError("partial UB inf not supported")
            if flag_LB == -1 and flag_UB == -1:
                raise ValueError("partial LB and UB inf not supported")
            if flag_LB == 1 and flag_UB != -1:
                G = matrix(np.eye(n))
                h = matrix(UB)
            if flag_UB == 1 and flag_LB != -1:
                G = matrix(-np.eye(n))
                h = matrix(-LB)

    solved = solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b,
                        kktsolver=kktsolver, options=opts)

    return solved
