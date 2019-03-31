import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR) # Remove tf warnings
import numpy as np
from snn.core.mlp_fn import *
from snn.core.extra_fn import *
from snn.core.data_fn import *
from snn.core.sgd import *
from snn.core import package_path
from snn.core.network import Network
import pickle # For saving python objects

class FC(Network):
    """ A Fully Connected neural network learning model """

    def __init__(self, X, Y, logging=True, layers=[784, 600, 10], scopes_list=['hidden1', 'output'], graph=tf.Graph(),
                 seed=11, initial_weights=None):
        # Initialize the tensorflow graph and session
        Network.__init__(self, X, Y, logging, layers, scopes_list, graph, seed)
        with self.graph.as_default():
            tf.set_random_seed(self.seed)
            np.random.seed(self.seed)
            self.sess = tf.Session(graph=self.graph)

            # Tensorflow placeholders
            self.x, self.y = self.create_placeholders()
            if initial_weights is None:
                self.yhat = multilayer_perceptron(self.x, self.layers, self.scopes_list)  # The base network
            else:  # Else, have to initialize according to the passed in weights
                self.yhat, params_list = multilayer_perceptron_init(self.x, self.layers, initial_weights,
                                                                    self.scopes_list)
            self.model_with_noise = MLP_withnoise
        return

    def get_model_weights(self):
        with self.graph.as_default():
            self.tf_init # Not called if it has already been called
            param_val_list = []
            for (scopename,n_in,n_out) in zip(self.scopes_list,self.layers[:-1],self.layers[1:]):
                with tf.variable_scope(scopename, reuse=True) as scope:
                    W_h = tf.get_variable('weights', shape=[n_in, n_out])
                    b_h = tf.get_variable('biases', shape=[n_out])
                    param_val_list.append(W_h.eval(session=self.sess))
                    param_val_list.append(b_h.eval(session=self.sess))
            return param_val_list

    def count_N_params(self):
        """
        For computing VC dimension bound with boundVCdim()
        """
        N = 0
        par_shapes = []
        for (n_in, n_out) in zip(self.layers[:-1], self.layers[1:]):
            N += n_in * n_out
            N += n_out
            par_shapes.append([n_in,n_out])
            par_shapes.append([n_out])
        return N, par_shapes

    def load_model_weights(self, params_mean_values):
        # Hacky way to Create a new graph and session for now, resets the default:
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.seed)
            np.random.seed(self.seed)
            self.sess = tf.Session(graph=self.graph)
            self.x, self.y = self.create_placeholders()
            self.yhat, param_tensor_list = multilayer_perceptron_init(self.x, self.layers, params_mean_values, self.scopes_list)
            self.force_tf_init() # Force initialize it
        return param_tensor_list

    def create_placeholders(self):
        # Recreate Placeholders for the optimization
        with self.graph.as_default():
            y = tf.placeholder(tf.float32, [None, self.layers[-1]])
            x = tf.placeholder(tf.float32, [None, self.layers[0]])
        return x, y
