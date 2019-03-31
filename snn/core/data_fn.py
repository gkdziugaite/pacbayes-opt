from __future__ import division, print_function, unicode_literals
import functools
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR) # Remove tf warnings
import numpy as np
from time import time
import os, shutil, random

from tensorflow.examples.tutorials.mnist import input_data

from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

NUM_CLASSES = 10
MNIST_DATA_DIR = "mnist/"
CIFAR_DATA_DIR = "CIFAR_data/"


def binarize_mnist_labels(labels):
    positive_mask = (labels > 4).astype(int).reshape(-1, 1)
    negative_mask = (labels <= 4).astype(int).reshape(-1, 1)
    mask = positive_mask - negative_mask
    print("mask: ", mask)
    return mask


def normalize_meanstd(images, axis=None, mean = None, std = None ):
    # axis param denotes axes along which mean & std reductions are to be performed
    if mean is None:
      mean = np.mean(images, axis=axis, keepdims=True)
      std = np.sqrt(((images - mean) ** 2).mean(axis=axis, keepdims=True))

    out = (images - mean) / std
    return out, mean, std


def load_mnist_data(data_dir = MNIST_DATA_DIR, one_hot=True):
    mnist = input_data.read_data_sets(data_dir, one_hot=one_hot)
    trainX, trainY = mnist.train.images, mnist.train.labels
    testX, testY = mnist.test.images, mnist.test.labels
    trainY = np.reshape(trainY.astype(float), [trainX.shape[0], NUM_CLASSES])
    testY = np.reshape(testY.astype(float), [testX.shape[0], NUM_CLASSES])
    return (trainX, trainY), (testX, testY)


def load_binary_mnist(data_dir = MNIST_DATA_DIR):
    mnist = input_data.read_data_sets(data_dir, one_hot=False)
    trainX, trainY = mnist.train.images, mnist.train.labels
    testX, testY = mnist.test.images, mnist.test.labels
    trainY = binarize_mnist_labels(trainY)
    testY = binarize_mnist_labels(testY)
    return (trainX, trainY), (testX, testY)


def load_cifar_data(data_dir = CIFAR_DATA_DIR, one_hot=True):
    """ Loads CIFAR10 data and whitens it using keras functions.
    TO DO: Look up data in data_dir! Right now it's being ignored.
    :param data_dir: directory where cifar data is stored (default  = 'cifar-10-batches-py').
    :param one_hot: BOOL, if True do one-hot encoding of the labels
    :return: returns (images, labels) for train and test data
    """

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
    x_train, xmean, xstd = normalize_meanstd(x_train, axis=0)
    x_test,_m,_s = normalize_meanstd(x_test, axis=0, mean=xmean, std=xstd)
    return (x_train, y_train), (x_test, y_test)


def next_batch(xx,yy, batchsize, idx):
    return [xx[idx*batchsize:(idx+1)*batchsize], yy[idx*batchsize:(idx+1)*batchsize]]


def shuffledata(xx,yy):
    ns = xx.shape[0]
    idx = random.sample(list(np.arange(ns)), ns)
    xx = xx[idx]
    yy = yy[idx]
    return xx,yy


def sort_corresponding_to_labels(trainX, trainY, label_order):
    """ Sorts the x and y such that they match the training order in label order. Tested. """
    trainY_labels = np.where(trainY > 0)[1]

    idxs = np.arange(0, len(trainY_labels))
    idmap = dict((id,pos) for pos,id in enumerate(label_order))
    l_ordered = [(x, y, _, _o) for _, _o, x, y in sorted(zip(idxs, trainY_labels, trainX, trainY), key=lambda x:idmap[x[1]])]
    newlabord = [l_ordered[i][3] for i in range(len(l_ordered))]
    x_ordered = np.array([l_ordered[i][0] for i in range(len(l_ordered))])
    y_ordered = np.array([l_ordered[i][1] for i in range(len(l_ordered))])

    return x_ordered, y_ordered, newlabord


def concat_labels(label_order):
    return np.hstack(label_order)


def label_indices(x, L):
    """ Returns the indices of data points with label L. TESTED."""
    labels = np.where(x > 0)[1]
    return np.where(labels == L)[0]


def sample_label(x, L):
    """ Samples a random instance with label L. Returns the index of the data point location. """
    idx = label_indices(x, L)
    return np.random.choice(idx)


def sample_label_from_dict(dd, L, size=None, replace=True):
    if isinstance(L, np.ndarray):
        return np.random.choice(dd[L[0]], size=size, replace=replace)
    else:
        return np.random.choice(dd[L], size=size, replace=replace)


def label(x):
    """ Returns the mnist number label corresponding to the array """
    if x.ndim == 1:
        return np.where(x>0)[0]
    else:
        return np.where(x>0)[1]


def idx_to_label(x, trainY):
    """ Returns a an array of labels given an array of indexes and training set """
    y_lab = label(trainY)
    return y_lab[x]
