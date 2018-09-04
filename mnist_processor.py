from __future__ import division
import os
from GAN import GAN
from tensorflow.contrib.tensorboard.plugins import projector
import time
import tensorflow as tf
import datetime
import PIL.Image
import sklearn.preprocessing
import numpy as np

from ops import *
from utils import *
import chainer

from chainer.datasets import TransformDataset
from chainercv.transforms import resize

def svhn():
    source, source_train= chainer.datasets.get_svhn()

    labels = []

    images = []

    for i in range(5000):

        print(i)

        image = source_train[i][0]

        label = source_train[i][1]

        if label >= 5:
            image = np.transpose(image, [1, 2, 0])

            images.append(image.astype(np.float32))

            labels.append(label - 5)


    images = np.array(images).astype(np.float32)
    labels = np.squeeze(np.array(labels)).astype(np.int32)

    print(labels[4])
    print(images[4])

    print(labels.shape)
    print(images.shape)

    print(labels[330])
    PIL.Image.fromarray((images[330] * 256).astype(np.int8), 'RGB').show()

    np.save('svhn_rgb_images_valid', images)
    np.save('svhn_rgb_labels_valid', labels)

    lb = sklearn.preprocessing.LabelBinarizer().fit([0, 1, 2, 3, 4])

    labels = lb.transform(labels).astype(np.float32)

    np.save('svhn_rgb_labels_onehot_valid', labels)


def mnist():

    target_train, target_test = chainer.datasets.get_mnist(ndim=3, rgb_format=True)

    def transform(in_data):
        img, label = in_data
        img = resize(img, (32, 32))
        return img, label

    source_ = TransformDataset(target_train, transform)


    source_train = TransformDataset(target_test, transform)

    labels = []

    images = []

    for i in range(10000):

        print(i)

        image = source_train[i][0]

        label = source_train[i][1]

        if label >= 5:
            image = np.transpose(image, [1, 2, 0])

            images.append(image.astype(np.float32))

            labels.append(label - 5)

    images = np.array(images).astype(np.float32)
    labels = np.squeeze(np.array(labels)).astype(np.int32)

    print(labels[4])
    print(images[4])

    print(labels.shape)
    print(images.shape)

    print(labels[330])
    PIL.Image.fromarray((images[330] * 256).astype(np.int8), 'RGB').show()

    np.save('mnist_rgb_images_test', images)
    np.save('mnist_rgb_labels_test', labels)

    lb = sklearn.preprocessing.LabelBinarizer().fit([0, 1, 2, 3, 4])

    labels = lb.transform(labels).astype(np.float32)

    np.save('mnist_rgb_labels_onehot_test', labels)

# svhn()

mnist()