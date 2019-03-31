from __future__ import division, print_function, unicode_literals
import functools
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR) # Remove tf warnings
import numpy as np
from time import time
import os, shutil, random

NUM_CLASSES = 10


def CNN_withnoise(images, param_placeholders, scopes_list,layers,params_mean_values, graph=tf.Graph(), trainable=True):
    with graph.as_default():

        param_tensor_list = []
        with tf.variable_scope(scopes_list[0]) as scope:
            kernel = variable_initializer('weights',
                                          [5, 5, 3, 64],
                                          tf.constant_initializer(params_mean_values[0]), trainable = trainable)
            conv = tf.nn.conv2d(images, kernel+param_placeholders[0], [1, 1, 1, 1], padding='SAME')
            biases = variable_initializer('biases', [64], tf.constant_initializer(params_mean_values[1]), trainable = trainable)
            pre_activation = tf.nn.bias_add(conv, biases+param_placeholders[1])
            conv1 = tf.nn.relu(pre_activation, name=scope.name)

        param_tensor_list.append(kernel)
        param_tensor_list.append(biases)
        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool1')
        # norm1
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm1')

        # conv2
        with tf.variable_scope(scopes_list[1]) as scope:
            kernel = variable_initializer('weights',
                                              [5, 5, 64, 64],
                                              tf.constant_initializer(params_mean_values[2]), trainable = trainable)
            conv = tf.nn.conv2d(norm1, kernel+param_placeholders[2], [1, 1, 1, 1], padding='SAME')
            biases = variable_initializer('biases', [64], tf.constant_initializer(params_mean_values[3]),  trainable = trainable)
            pre_activation = tf.nn.bias_add(conv, biases+param_placeholders[3], name=scope.name)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)

        param_tensor_list.append(kernel)
        param_tensor_list.append(biases)
        # norm2
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                              name='norm2')
        # pool2
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1], padding='SAME', name='pool2')
        # local3
        with tf.variable_scope(scopes_list[2]) as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            reshape = tf.reshape(pool2, [-1, 4096])
            dim = reshape.get_shape()[1].value
            weights = variable_initializer('weights', [dim, 384],
                                           tf.constant_initializer(params_mean_values[4]), trainable = trainable)
            biases = variable_initializer('biases', [384], tf.constant_initializer(params_mean_values[5]), trainable = trainable)
            local3 = tf.nn.relu(tf.matmul(reshape, weights+param_placeholders[4]) + biases+param_placeholders[5], name=scope.name)

        param_tensor_list.append(weights)
        param_tensor_list.append(biases)
        # local4
        with tf.variable_scope(scopes_list[3]) as scope:
            weights = variable_initializer('weights', [384, 192],
                                               tf.constant_initializer(params_mean_values[6]), trainable = trainable)
            biases = variable_initializer('biases', [192], tf.constant_initializer(params_mean_values[7]), trainable = trainable)
            local4 = tf.nn.relu(tf.matmul(local3, weights+param_placeholders[6]) + biases + param_placeholders[7], name=scope.name)

        param_tensor_list.append(weights)
        param_tensor_list.append(biases)

        with tf.variable_scope(scopes_list[4]) as scope:
            weights = variable_initializer('weights', [192, NUM_CLASSES],
                                               tf.constant_initializer(params_mean_values[8]), trainable = trainable)
            biases = variable_initializer('biases', [NUM_CLASSES],
                                          tf.constant_initializer(params_mean_values[9]), trainable = trainable)
            softmax_linear = tf.add(tf.matmul(local4, weights+param_placeholders[8]), biases+param_placeholders[9], name=scope.name)

        param_tensor_list.append(weights)
        param_tensor_list.append(biases)

        return softmax_linear, param_tensor_list


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

def convolutional_net(images, scopes_list = ['conv1','conv2','local3','local4','softmax_linear']):
  """Build the CIFAR-10 model.

  Args:
    x: Images placeholder.

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope(scopes_list[0]) as scope:
    kernel = variable_initializer('weights',
                                  [5, 5, 3, 64],
                                  tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = variable_initializer('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope(scopes_list[1]) as scope:
    kernel = variable_initializer('weights',
                                  [5, 5, 64, 64],
                                  tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = variable_initializer('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope(scopes_list[2]) as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [-1, 4096])
    dim = reshape.get_shape()[1].value
    weights = variable_initializer('weights', [dim, 384],
                                   tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32))
    biases = variable_initializer('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

  # local4
  with tf.variable_scope(scopes_list[3]) as scope:
    weights = variable_initializer('weights', [384, 192],
                                   tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32))
    biases = variable_initializer('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

  with tf.variable_scope(scopes_list[4]) as scope:
    weights = variable_initializer('weights', [192, NUM_CLASSES],
                                   tf.truncated_normal_initializer(stddev=1/192.0, dtype=tf.float32) )
    biases = variable_initializer('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

  return softmax_linear


def convolutional_net_init(images, params_mean_values=None,
                           scopes_list=['conv1', 'conv2', 'local3', 'local4', 'softmax_linear']):
    """
    Build CIFAR10 model with provided initial values
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    # conv1
    param_tensor_list = []
    with tf.variable_scope(scopes_list[0]) as scope:
        kernel = variable_initializer('weights',
                                      [5, 5, 3, 64],
                                      tf.constant_initializer(params_mean_values[0]))
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_initializer('biases', [64], tf.constant_initializer(params_mean_values[1]))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
    #  _activation_summary(conv1)
    param_tensor_list.append(kernel)
    param_tensor_list.append(biases)
    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                            padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # conv2
    with tf.variable_scope(scopes_list[1]) as scope:
        kernel = variable_initializer('weights',
                                          [5, 5, 64, 64],
                                          tf.constant_initializer(params_mean_values[2]))
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_initializer('biases', [64], tf.constant_initializer(params_mean_values[3]))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
    param_tensor_list.append(kernel)
    param_tensor_list.append(biases)
    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    # local3
    with tf.variable_scope(scopes_list[2]) as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2, [-1, 4096])
        dim = reshape.get_shape()[1].value
        weights = variable_initializer('weights', [dim, 384],
                                       tf.constant_initializer(params_mean_values[4]))
        biases = variable_initializer('biases', [384], tf.constant_initializer(params_mean_values[5]))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    param_tensor_list.append(weights)
    param_tensor_list.append(biases)
    # local4
    with tf.variable_scope(scopes_list[3]) as scope:
        weights = variable_initializer('weights', [384, 192],
                                           tf.constant_initializer(params_mean_values[6]))
        biases = variable_initializer('biases', [192], tf.constant_initializer(params_mean_values[7]))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

    param_tensor_list.append(weights)
    param_tensor_list.append(biases)

    with tf.variable_scope(scopes_list[4]) as scope:
        weights = variable_initializer('weights', [192, NUM_CLASSES],
                                           tf.constant_initializer(params_mean_values[8]))
        biases = variable_initializer('biases', [NUM_CLASSES],
                                      tf.constant_initializer(params_mean_values[9]))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    # _activation_summary(softmax_linear)
    param_tensor_list.append(weights)
    param_tensor_list.append(biases)

    return softmax_linear, param_tensor_list


def weight_diff(w1, w2):
    """ Calculates the array of differences between the weights in arrays """
    # Expand and flatten arrays
    _w1 = np.hstack([x.flatten() for x in w1])
    _w2 = np.hstack([x.flatten() for x in w2])
    return _w1 - _w2


def l2_norm(w1, w2):
    return np.linalg.norm(weight_diff(w1, w2))
