import numpy as np

from utils import convert_strings_into_integers, define_binary_label
from CPMMC import CPMMC


def toy_loader(path):
    with open(path, 'r') as f:
        data = f.readlines()

    T_data = convert_strings_into_integers(data).transpose()
    label = T_data[-1]

    # import ipdb
    # ipdb.set_trace()

    bs = T_data.shape[1] - 1
    # assuming all points have same dims
    dim = T_data.shape[0]

    # [:-1] in T_data for removing label dimension
    return T_data.transpose()[:,:-1], label, bs, dim

def main(path):

    # binary clustering
    binary = [2, 4]

    # for toy dataset max index is 9 min index is 0
    data, label, bs, dim = toy_loader(path)

    # generate binary label
    defined_label1, defined_label2 = define_binary_label(label, binary)
    num_total_samples = len(defined_label1) + len(defined_label2)
    label = np.ones(num_total_samples, dtype='int64')
    label[len(defined_label2):-1] = -1
    index = np.concatenate((defined_label1, defined_label2))
    # index.sort(axis=0)
    # training_data = data[np.concatenate((defined_label1, defined_label2))]
    training_data = data[index]
    return training_data, label

if __name__ == '__main__':
    PATH = './optdigits.txt'
    data, label = main(path=PATH)

    #convert into (1, 1) numpy array for further deployment
    C = np.array([[0.01]])
    l = np.array([[10]])
    b_0 = np.array([[0]])
    xi_0 = np.array([[0.5]])

    epsilon = 10

    MMC = CPMMC(data=data, anns=label,
                C=C, epsilon=epsilon, l=l,
                b_0=b_0, xi_0=xi_0)

    acc = MMC()
