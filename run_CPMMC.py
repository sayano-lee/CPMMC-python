import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

from utils import convert_strings_into_integers, define_binary_label
from CPMMC import CPMMC

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
#fetch original mnist dataset
from sklearn.datasets import fetch_mldata

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


def opt_dataset(path):

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

def get_mnist_nums_idx(images, label, classes):
    label = label.astype('uint8')
    old_lbl = 0
    split = []
    for cnt, lbl in enumerate(label):
        lbl = int(lbl)
        if lbl == old_lbl:
            continue
        else:
            old_lbl = lbl
            split.append(cnt)
    data1 = images[split[classes[0]-1]:split[classes[0]]]
    data2 = images[split[classes[1]-1]:split[classes[1]]]
    label1 = np.ones(data1.shape[0])
    label2 = - np.ones(data2.shape[0])

    return np.concatenate((data1, data2)), np.concatenate((label1, label2))


if __name__ == '__main__':

    # ===========|  optdigits  |=================
    PATH = './optdigits.txt'
    data, label = opt_dataset(path=PATH)

    # MMC algorithm
    #convert into (1, 1) numpy array for further deployment
    C = np.array([[15]])
    l = np.array([[1]])
    CountVioCoef = 1.15e-2
    b_0 = np.array([[0]])
    xi_0 = np.array([[0.5]])
    dim = 784
    epsilon = 10

    MMC = CPMMC(dim=dim, C=C, l=l, CountViolateCoef=CountVioCoef,
                epsilon=epsilon,
                b_0=b_0, xi_0=xi_0)

    '''
    acc = MMC(data, label)
    print("accuracy of MMC is {:.4f}".format(float(acc[0])/len(acc[1])))
    '''

    # SVM algorithm

    clf = svm.SVC(C=1, gamma=0.01)

    '''
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3)
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print("accuracy of svm is {:.4f}".format(score))
    '''


    # ===========|  mnist  |=================
    # it creates mldata folder in your root project folder
    mnist = fetch_mldata('MNIST original', data_home='./')

    #minist object contains: data, COL_NAMES, DESCR, target fields
    #you can check it by running
    mnist.keys()

    #data field is 70k x 784 array, each row represents pixels from 28x28=784 image
    images = (mnist.data) / 255.0
    targets = mnist.target

    # MMC algorithm for mnist
    classes = [1, 7]
    x, y = get_mnist_nums_idx(images, targets, classes)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    acc = MMC(x_train, y_train)
    print("accuracy of MMC is {:.4f}".format(float(acc[0])/len(acc[1])))

    # svm algorithm for mnist
    # clf.fit(x_train, y_train)
    # score = clf.score(x_test, y_test)
    # print("accuracy of svm is {:.4f}".format(score))

    import ipdb
    ipdb.set_trace()