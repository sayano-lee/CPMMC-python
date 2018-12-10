import numpy as np
# import math
from CCCP_MMC_dual import CCCP_MMC_dual

class CPMMC(object):

    def __init__(self, **kwargs):

        self.data = kwargs['data']
        self.ann = kwargs['anns']


        self.bs = self.data.shape[0]
        self.dim = self.data.shape[1]

        self.C = kwargs['C']
        self.epsilon = kwargs['epsilon']
        self.W = []
        self.l = kwargs['l']

        self.omega_0 = 0.003 * np.ones((self.dim, 1))
        self.b_0 = 0
        self.xi_0 = 0.5


    def __call__(self):

        constraint = np.zeros((self.bs, 1))
        for i in range(self.bs):

            intermediate = self.omega_0.transpose().dot(self.data[i,:][:,np.newaxis])

            CountViolate = abs(intermediate) + self.b_0
            if CountViolate < 1:
                constraint[i] = 1
            else:
                constraint[i] = 0

        self.W = constraint.transpose()

        continue_flag = True

        while(continue_flag):

            omega, b, xi = CCCP_MMC_dual(omega_0=self.omega_0, b_0=self.b_0,
                                         xi_0=self.xi_0, C=self.C,
                                         W=self.W, l=self.l,
                                         data=self.data)

            constraint = np.zeros((self.dim, 1))
            SumQuit = 0

            for i in range(self.dim):
                CountViolate = abs(omega.transpose().dot(self.data[i,:])+b)
                if CountViolate<1:
                    constraint[i] = 1
                    SumQuit = SumQuit + constraint[i] - constraint[i] * CountViolate
                else:
                    constraint[i] = 0

            SumQuit = SumQuit / self.dim

            # end loop
            if SumQuit <= xi * (1 + self.epsilon):
                continue_flag = False
            else:
                import ipdb
                ipdb.set_trace()
            import ipdb
            ipdb.set_trace()

