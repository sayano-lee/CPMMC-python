import numpy as np
# import quadprog as qp
from cvxopt import solvers, matrix
import quadprog

from solver_wrapper import solve_qp

def CCCP_MMC_dual(**kwargs):

    omega_0 = kwargs['omega_0']
    b_0 = kwargs['b_0']
    xi_0 = kwargs['xi_0']
    C = kwargs['C']
    W = kwargs['W']
    l = kwargs['l']
    data = kwargs['data'].transpose()

    constraint_dim, data_dim = W.shape[0], W.shape[1]
    dim, tmp = omega_0.shape[0], omega_0.shape[1]

    # omega_old = omega_0
    # b_old = b_0
    # xi_old = xi

    f_val = 0.5 * omega_0.transpose() * omega_0 + C * xi_0

    continue_flag = True
    per_quit = 0.01

    # iter = 0
    c_k = W.mean(axis=1)[np.newaxis,:]
    # s_k = np.zeros((constraint_dim, 1))
    # z_k = np.zeros((dim, constraint_dim))
    x_k = np.sum(data, axis=1)[:,np.newaxis]

    while(continue_flag):
        tmp_z_k = np.zeros((dim, data_dim))
        tmp_s_k = np.zeros((data_dim, 1))

        for i in range(data_dim):
            tmp_s_k[i] = np.sign(omega_0.transpose().dot(data[:,i][:,np.newaxis])+b_0)
            tmp_z_k[:,i] = tmp_s_k[i] * data[:,i]

        s_k = W.dot(tmp_s_k) / data_dim
        z_k = tmp_z_k.dot(W.transpose()) / data_dim

        # import ipdb
        # ipdb.set_trace()
        x_mat = np.concatenate((z_k, -x_k, x_k), axis=1)
        HQP = x_mat.transpose().dot(x_mat)
        # HQP = np.array([[1.0,2.0,3.0],[2.0,5.0,8.0],[3.0,8.0,9.0]])
        # fQP = [-c_k, l, l]
        fQP = np.concatenate((-c_k, l, l), axis=0)

        suffix = np.array([[0, 0]])    #shape (1,2)
        prefix = np.ones((1, constraint_dim))
        # AQP = np.append(np.ones((1, constraint_dim)), suffix, axis=0)    #shape (1, c_dim+2)
        AQP = np.concatenate((prefix, suffix), axis=1)    #shape (1, dim(prefix) + 2)
        bQP = C

        data_dim_arr = np.array([[data_dim]])
        Aeq = np.concatenate((-s_k.transpose(), data_dim_arr, -data_dim_arr),axis=1)
        beq = np.array([[0]], dtype=float)



        # [XQP, fVal, exitFlag] = quadprog(HQP, fQP, AQP, bQP, Aeq, beq, LB, UB, [], ops)

        argus = [matrix(i) for i in [HQP, fQP, AQP, bQP, Aeq, beq]]

        # solved = solvers.qp(*arguments)
        import ipdb
        ipdb.set_trace()

        solved = solve_qp(HQP, fQP, AQP, bQP, Aeq, beq)

        solved = quadprog.solve_qp(HQP, fQP, AQP, bQP, Aeq, beq)
        #solved = solvers.qp(P=HQP, q=fQP)


