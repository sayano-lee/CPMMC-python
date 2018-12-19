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


def find_label_index(label1, label2):
    label = np.ones(len(label1) + len(label2), dtype="int64")

    index = np.concatenate((label1, label2))
    index.sort()

    # find index in label2, set to -1
    for cnt, idx in enumerate(index):
        if idx in label2:
            label[cnt] = -1
    return index, label


def main(path):

    # binary clustering
    binary = [2,8]

    # for toy dataset max index is 9 min index is 0
    data, label, bs, dim = toy_loader(path)

    # generate binary label
    defined_label1, defined_label2 = define_binary_label(label, binary)
    index, label = find_label_index(defined_label1, defined_label2)
    # index.sort(axis=0)
    # training_data = data[np.concatenate((defined_label1, defined_label2))]
    training_data = data[index]
    return training_data, label

if __name__ == '__main__':
    PATH = './optdigits.txt'
    data, label = main(path=PATH)
    import ipdb
    ipdb.set_trace()

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
    print("accuracy is {:.4f}".format(acc))
