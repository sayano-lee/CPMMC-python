import numpy as np
# import math
from CCCP_MMC_dual import CCCP_MMC_dual

class CPMMC(object):

    def __init__(self, **kwargs):

        """

        C, l, omega_0 need to be fine tuned

        """

        # self.data = kwargs['data']
        # self.ann = kwargs['anns']

        dim = kwargs['dim']

        # self.bs = self.data.shape[0]
        # self.dim = self.data.shape[1]

        self.C = kwargs['C']
        self.epsilon = kwargs['epsilon']
        self.W = []
        self.l = kwargs['l']

        self.omega_0 = 0.001 * np.ones((dim, 1))
        self.b_0 = 0
        self.xi_0 = 0.5


    def __call__(self, data, label):

        bs = data.shape[0]
        # dim = data.shape[1]

        constraint = np.zeros((bs, 1))

        for i in range(bs):

            intermediate = self.omega_0.transpose().dot(data[i,:][:,np.newaxis])
            CountViolate = abs(intermediate) + self.b_0

            if CountViolate < 1:
                constraint[i] = 1
            else:
                constraint[i] = 0

        self.W = constraint.transpose()

        continue_flag = True

        cnt = 0
        while(continue_flag):

            cnt += 1

            omega, b, xi = CCCP_MMC_dual(omega_0=self.omega_0, b_0=self.b_0,
                                         xi_0=self.xi_0, C=self.C,
                                         W=self.W, l=self.l,
                                         data=data)

            constraint = np.zeros((bs, 1))
            SumQuit = 0

            for i in range(bs):
                CountViolate = abs(omega.transpose().dot(data[i,:])+b)
                if CountViolate<1:
                    constraint[i] = 1
                    SumQuit = SumQuit + constraint[i] - constraint[i] * CountViolate
                else:
                    constraint[i] = 0

            SumQuit = SumQuit / bs

            # when to end loop
            if SumQuit <= xi * (1 + self.epsilon):
                continue_flag = False
            else:
                self.W = np.concatenate((self.W, constraint.transpose()), axis=0)
                self.omega_0 = omega
                self.b_0 = b
                self.xi_0 = xi

        count = 0
        # predicted_label = np.sign(omega.transpose().dot(self.data.transpose()) + b)
        predicted_label = np.sign(data.dot(omega) + b).astype('int')

        for i in range(bs):
            if predicted_label[i] == label[i]:
                count += 1

        return float(count) / bs, predicted_label

