import numpy as np


## FIX ME: USED ONLY FOR TOY DATAS
def convert_string_into_integer(line):
    return list(map(int, line.strip('\n').split(',')))


def convert_strings_into_integers(data):
    lines = []
    for line in data:
        lines.append(convert_string_into_integer(line))
    return np.array(lines)


def define_binary_label(label, idx):
    indices1 = np.where(label == idx[0])
    indices2 = np.where(label == idx[1])
    return indices1[0], indices2[0]