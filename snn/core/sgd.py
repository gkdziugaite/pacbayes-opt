from snn.core.mlp_fn import *
from snn.core.extra_fn import *
from snn.core.data_fn import *
import time
import math

import numpy as np # TODO: This was not imported


def label_indices(x, L):
    """ Returns the indices of data points with label L. TESTED."""
    labels = np.where(x > 0)[1]
    return np.where(labels == L)[0]


class SGD:
    """ Specifies the variant of the SGD algorithm used, and takes in
    epochs and iterations and outputs the next batch of data points to train """

    def __init__(self, X, Y, total_epochs=20, batch_size=100, seed=11):
        self.total_epochs = total_epochs
        self.trainX = X
        self.trainY = Y
        self.Nsamples = X.shape[0]
        self.batch_size = batch_size
        self.past_epoch = 1
        self.index_set = range(0, self.trainX.shape[0])

        # Set random seed
        self.seed = seed
        np.random.seed(self.seed)

    def __del__(self):
        return

    def epoch_train_set(self, epoch=0): # TODO: Rewrite function to only return the shuffled indices
        """ Produces the training set to be used for every epoch."""

        index_set_shuffle = self.index_set
        if epoch != 0: # The first epoch should not reshuffle index_set
            index_set_shuffle = np.random.permutation(self.index_set) # Reshuffle indices
        return index_set_shuffle
