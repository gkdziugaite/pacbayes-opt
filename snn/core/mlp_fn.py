from __future__ import division, print_function, unicode_literals
import functools
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR) # Remove tf warnings
import numpy as np
from time import time
import os, shutil, random


def MLP_withnoise(x, param_placeholders, scopes_list,layers,params_mean_values, graph=tf.Graph(), trainable=True):
    with graph.as_default():
        param_list = []
        with tf.variable_scope(scopes_list[0]) as scope:
          W_mean = variable_initializer('weights', [layers[0], layers[1]], tf.constant_initializer(params_mean_values[0]), trainable=trainable)
          b_mean = variable_initializer('biases', [layers[1]], tf.constant_initializer(params_mean_values[1]), trainable=trainable)
          next_layer = tf.add(tf.matmul(x,param_placeholders[0]+W_mean), param_placeholders[1]+b_mean)
          param_list.append(W_mean)
          param_list.append(b_mean)
        for (w,b,scopename,n_in,n_out,wmean,bmean) in zip(param_placeholders[2::2],
                            param_placeholders[3::2],scopes_list[1:],
                            layers[1:-1],layers[2:], params_mean_values[2::2], params_mean_values[3::2]):
            with tf.variable_scope(scopename) as scope:
              W_mean = variable_initializer('weights', [n_in, n_out], tf.constant_initializer(wmean), trainable=trainable)
              b_mean = variable_initializer('biases', [n_out], tf.constant_initializer(bmean), trainable=trainable)
              next_layer = tf.nn.relu(next_layer)
              next_layer = tf.add(tf.matmul(next_layer,w+W_mean), b+b_mean)
              param_list.append(W_mean)
              param_list.append(b_mean)
        return next_layer,param_list


def lazy_property(function):
    """ Create a property such that defining model classes is easier with tf """
    attribute = '_cache_' + function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


def variable_initializer(name, shape, initializer, trainable=True):
    return tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32, trainable=trainable)


def multilayer_perceptron(x, layers, scopes_list = ['hidden1','output']):
    # Fully connected feedforward network with RELU activation
    n_in= layers[0]
    n_out = layers[1]
    with tf.variable_scope(scopes_list[0]) as scope:
      weights = variable_initializer('weights', [n_in,n_out],
                                  tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32))
      biases = variable_initializer('biases', [n_out], tf.constant_initializer(0.1))
      next_layer = tf.add(tf.matmul(x,weights), biases)
      tf.summary.histogram("W1",weights)
      tf.summary.histogram("b1",biases)
    for (scopename,n_in,n_out) in zip(scopes_list[1:],layers[1:-1],layers[2:]):
        with tf.variable_scope(scopename) as scope:
          weights = variable_initializer('weights', [n_in,n_out],
                                      tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32))
          biases = variable_initializer('biases', [n_out], tf.constant_initializer(0))
          next_layer = tf.nn.relu(next_layer)
          next_layer = tf.add(tf.matmul(next_layer,weights), biases)
          tf.summary.histogram("W1",weights)
          tf.summary.histogram("b1",biases)

    return next_layer


def multilayer_perceptron_init(x, layers, params_mean_values, scopes_list = ['hidden1','output']):
    # Fully connected feedforward network with RELU activation
    param_list = []
    n_in= layers[0]
    n_out = layers[1]
    with tf.variable_scope(scopes_list[0]) as scope:
      weights = variable_initializer('weights', [n_in,n_out],
                                  tf.constant_initializer(params_mean_values[0]))
      biases = variable_initializer('biases', [n_out], tf.constant_initializer(params_mean_values[1]))
      next_layer = tf.add(tf.matmul(x,weights), biases)
      tf.summary.histogram("W1",weights)
      tf.summary.histogram("b1",biases)
      param_list.append(weights)
      param_list.append(biases)
    for (scopename,n_in,n_out, wmean, bmean) in zip(scopes_list[1:],layers[1:-1],layers[2:], params_mean_values[2::2], params_mean_values[3::2]):
        with tf.variable_scope(scopename) as scope:
          weights = variable_initializer('weights', [n_in,n_out],
                                      tf.constant_initializer(wmean))
          biases = variable_initializer('biases', [n_out], tf.constant_initializer(bmean))
          next_layer = tf.nn.relu(next_layer)
          next_layer = tf.add(tf.matmul(next_layer,weights), biases)
          tf.summary.histogram("W1",weights)
          tf.summary.histogram("b1",biases)
          param_list.append(weights)
          param_list.append(biases)

    return next_layer, param_list


def weight_diff(w1, w2):
    """ Calculates the array of differences between the weights in arrays """
    # Expand and flatten arrays
    _w1 = np.hstack([x.flatten() for x in w1])
    _w2 = np.hstack([x.flatten() for x in w2])
    return _w1 - _w2


def l2_norm(w1, w2):
    return np.linalg.norm(weight_diff(w1, w2))
