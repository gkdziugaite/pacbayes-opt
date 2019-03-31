from __future__ import division, print_function, unicode_literals
import functools
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR) # Remove tf warnings
import numpy as np
from time import time
import os, shutil, random


def generate_noise(layer_shapes):
    noise_list = []
    for l_shape in layer_shapes:
        noise_list.append(np.random.normal(size=l_shape))
    return noise_list


def generate_zero_noise(layer_shapes):
    noise_list = []
    for l_shape in layer_shapes:
        noise_list.append(np.zeros(l_shape))
    return noise_list


def margin_loss(yhat, y):
    mls = []
    for row in range(len(y)):
        j = np.argmax(y[row])
        mls.append(yhat[row][j] - max([i for i in range(len(yhat[row])) if i != j]))
    return mls


def count_MLP_params(layers):
    """
    For computing VC dimension bound with boundVCdim()
    """
    N = 0
    for (n_in,n_out) in zip(layers[:-1],layers[1:]):
        N += n_in*n_out
        N += n_out
    return N


def KLdiv(pbar,p):
    return pbar * np.log(pbar/p) + (1-pbar) * np.log((1-pbar)/(1-p))


def KLdiv_prime(pbar,p):
    return (1-pbar)/(1-p) - pbar/p


def Newt(p,q,c):
    newp = p - (KLdiv(q,p) - c)/KLdiv_prime(q,p)
    return newp


def approximate_BPAC_bound(train_accur, B_init, niter=5):
    B_RE = 2* B_init **2
    A = 1-train_accur
    B_next = B_init+A
    if B_next>1.0:
        return 1.0
    for i in range(niter):
        B_next = Newt(B_next,A,B_RE)
    return B_next


def hoeffdingbnd(M,delta):
    eps = np.sqrt(np.log(2/delta)/M)
    return eps


def SamplesConvBound(train_error=0.028,M=1000,delta=0.01,p_init = None, niter = 5):
    c =  np.log(2/delta)/M
    if p_init is None:
        p_init = hoeffdingbnd(M,delta)
        print("Hoeffding's error", p_init)
    p_next = p_init+train_error
    for i in range(niter):
        p_next = Newt(p_next,train_error,c)
    print("Chernoff's error", p_next-train_error)
    return p_next-train_error


def next_batch(xx,yy, batchsize, idx):
    return [xx[idx*batchsize:(idx+1)*batchsize], yy[idx*batchsize:(idx+1)*batchsize]]


def shuffledata(xx,yy):
    ns = xx.shape[0]
    idx = random.sample(list(np.arange(ns)), ns)
    xx = xx[idx,:]
    yy = yy[idx,:]
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


def create_label_dictionary(yy):
    """ Creates the label dictionary used to sample data points within the priors dataset"""
    labels = np.where(yy > 0)[1]
    idxs = range(0, yy.shape[0])
    dict = {}
    for x, y in zip(idxs, labels):
        dict.setdefault(y,[]).append(x)
    return dict
