import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR) # Remove tf warnings
import numpy as np
from snn.core.cnn_fn import *
from snn.core.extra_fn import *
from snn.core.data_fn import *
from snn.core.sgd import *
from snn.core import package_path
import pickle # For saving python objects
from snn.core.network import Network
from snn.core import config

# GPU Related imports
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True

IMAGE_SIZE = 32


class CNN(Network):
    """ A network model """

    def __init__(self, X, Y, logging=True, layers = None, scopes_list=['conv1', 'conv2', 'local3', 'local4', 'softmax_linear'],
                 graph=tf.Graph(), seed=11, initial_weights=None):
        # Initialize the tensorflow graph and session

        Network.__init__(self, X, Y, logging, layers, scopes_list, graph, seed) # Initialize according to the network class

        with self.graph.as_default():
            tf.set_random_seed(seed)
            np.random.seed(seed)
            self.sess = tf.Session(graph=self.graph, config=config)

            # Tensorflow placeholders
            self.x, self.y = self.create_placeholders()

            if initial_weights is None:
                self.yhat = convolutional_net(self.x, self.scopes_list)  # The base network
            else:  # Else, have to initialize according to the passed in weights
                self.yhat, params_list = convolutional_net_init(self.x, initial_weights, self.scopes_list)

        self.model_with_noise = CNN_withnoise
        return

    def get_model_weights(self):

        with self.graph.as_default():
            self.tf_init # Not called if it has already been called
            param_val_list = []
            for (scopename) in (self.scopes_list):
                with tf.variable_scope(scopename, reuse=True) as scope:
                  W_h = tf.get_variable('weights')
                  b_h = tf.get_variable('biases')
                  param_val_list.append(W_h.eval(session=self.sess))
                  param_val_list.append(b_h.eval(session=self.sess))
            return param_val_list

    def count_N_params(self):
        try:
            return self.N_params # Always return the saved instantiation when you can
        except: # Else, calculate it again (should not be calculated again during the PAC Bound)
            N = 0
            par_shapes = []
            with self.graph.as_default():
              for scopename in self.scopes_list:
                with tf.variable_scope(scopename):
                    vals = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             scope=scopename)
                    for val in vals:
                      shape = val.get_shape().as_list()
                      par_shapes.append(shape)
                      N += np.prod(shape)

            self.N_params = N, par_shapes
            return self.N_params # Cast nparams to int or else what is returned is a tf dimension object

    def load_model_weights(self, params_mean_values):
        # Hacky way to Create a new graph and session for now, resets the default:
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.seed)
            np.random.seed(self.seed)
            self.sess = tf.Session(graph=self.graph, config=config)
            self.x, self.y = self.create_placeholders()
            self.yhat, param_tensor_list = convolutional_net_init(self.x, params_mean_values, self.scopes_list)
            self.force_tf_init()  # Force initialize it
        return param_tensor_list

    def create_placeholders(self):
        # Recreate Placeholders for the optimization
        batch_size = None
        with self.graph.as_default():
            y = tf.placeholder(tf.float32, [batch_size, self.layers[-1]])
            x = tf.placeholder(tf.float32, [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
        return x, y
