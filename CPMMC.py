import numpy as np


class CPMMC(object):

    def __init__(self, **kwargs):

        self.data = kwargs['training_data']
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


        import ipdb
        ipdb.set_trace()
