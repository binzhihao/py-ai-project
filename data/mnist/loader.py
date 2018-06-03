import os
import struct
import numpy as np
import matplotlib.pyplot as plt

module_path = os.path.dirname(__file__)


def load_mnist(path=module_path, kind='train'):
    """
    根据http://yann.lecun.com/exdb/mnist/对数据集的描述进行解析
    参考https://blog.csdn.net/simple_the_best/article/details/75267863
    :param path: file path
    :param kind: 'train' or 'test'
    :return: images is n x m ndarray where n is count of images, m is 784(28 X 28) each value means gray scale varies
        from 0 to 255.
    """
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as f_label:
        magic, n = struct.unpack('>II', f_label.read(8))
        labels = np.fromfile(f_label, dtype=np.uint8)

    with open(images_path, 'rb') as f_image:
        magic, num, rows, cols = struct.unpack('>IIII', f_image.read(16))
        images = np.fromfile(f_image, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def show_mnist(x_train, y_train):
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)

    ax = ax.flatten()

    for i in range(10):
        img = x_train[y_train == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
