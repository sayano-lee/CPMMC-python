import numpy as np
from utils import convert_strings_into_integers, define_binary_label


def toy_loader(path):
    with open(path, 'r') as f:
        data = f.readlines()

    T_data = convert_strings_into_integers(data).transpose()
    label = T_data[-1]

    bs = T_data.shape[1]
    # assuming all points have same dims
    dim = T_data.shape[0]

    # [:-1] in T_data for removing label dimension
    return T_data.transpose()[:-1], label, bs, dim

def main(path):


    # binary clustering
    binary = [0, 1]

    # for toy dataset max index is 9 min index is 0
    data, label, bs, dim = toy_loader(path)

    # generate binary label
    defined_label1, defined_label2 = define_binary_label(label, binary)
    num_total_samples = len(defined_label1) + len(defined_label2)
    label = np.ones(num_total_samples, dtype='int64')
    label[len(defined_label2):-1] = -1

    training_data = data[np.concatenate((defined_label1, defined_label2))]

    import ipdb
    ipdb.set_trace()


if __name__ == '__main__':
    PATH = './optdigits.txt'
    main(path=PATH)