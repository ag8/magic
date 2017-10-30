# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np

import tensorflow as tf

from constants import FLAGS

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))
from affinity.models.magic_autoencoder.utils.utils import *


slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(stddev=stddev)


def cifarnet(images, num_classes=10, is_training=False,
             dropout_keep_prob=0.5,
             prediction_fn=slim.softmax,
             scope='CifarNet'):
    """Creates a variant of the CifarNet model.
    Note that since the output is a set of 'logits', the values fall in the
    interval of (-infinity, infinity). Consequently, to convert the outputs to a
    probability distribution over the characters, one will need to convert them
    using the softmax function:
          logits = cifarnet.cifarnet(images, is_training=False)
          probabilities = tf.nn.softmax(logits)
          predictions = tf.argmax(logits, 1)
    Args:
      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.
    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    end_points = {}

    with tf.variable_scope(scope, 'CifarNet', [images, num_classes]):
        net = slim.conv2d(images, 64, [5, 5], scope='conv1')
        end_points['conv1'] = net
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
        end_points['pool1'] = net
        net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
        net = slim.conv2d(net, 64, [5, 5], scope='conv2')
        end_points['conv2'] = net
        net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
        end_points['pool2'] = net
        net = slim.flatten(net)
        end_points['Flatten'] = net
        net = slim.fully_connected(net, 384, scope='fc3')
        end_points['fc3'] = net
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout3')
        net = slim.fully_connected(net, 192, scope='fc4')
        end_points['fc4'] = net
        logits = slim.fully_connected(net, num_classes,
                                      biases_initializer=tf.zeros_initializer(),
                                      weights_initializer=trunc_normal(1 / 192.0),
                                      weights_regularizer=None,
                                      activation_fn=None,
                                      scope='logits')

        end_points['Logits'] = logits
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

    return logits, end_points


cifarnet.default_image_size = 32


def cifarnet_arg_scope(weight_decay=0.004):
    """Defines the default cifarnet argument scope.
    Args:
      weight_decay: The weight decay to use for regularizing the model.
    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
            [slim.conv2d],
            weights_initializer=tf.truncated_normal_initializer(stddev=5e-2),
            activation_fn=tf.nn.relu):
        with slim.arg_scope(
                [slim.fully_connected],
                biases_initializer=tf.constant_initializer(0.1),
                weights_initializer=trunc_normal(0.04),
                weights_regularizer=slim.l2_regularizer(weight_decay),
                activation_fn=tf.nn.relu) as sc:
            return sc


def net(images, num_fully_connected_layers=3):
    print("Images shape: "),
    print(images.get_shape())

    locks, keys = split(images)

    _, end_points_l = cifarnet(locks, scope='LockCifarNet')
    lock_feature_vector = end_points_l['fc4']

    _, end_points_k = cifarnet(keys, scope='KeyCifarNet')
    key_feature_vector = end_points_k['fc4']

    print("Lock feature vector shape: "),
    print(lock_feature_vector.get_shape())

    print("Key feature vector shape: "),
    print(key_feature_vector.get_shape())


    # Reshape into long tensors
    # locks = tf.reshape(locks, shape=[FLAGS.BATCH_SIZE, FLAGS.IMAGE_SIZE * FLAGS.IMAGE_SIZE])
    # keys = tf.reshape(keys, shape=[FLAGS.BATCH_SIZE, FLAGS.IMAGE_SIZE * FLAGS.IMAGE_SIZE])

    # print("Locks shape: "),
    # print(locks.get_shape())
    # print("Keys shape: "),
    # print(keys.get_shape())

    # 24x192
    elementwise_multiplied = tf.multiply(lock_feature_vector, key_feature_vector)

    if num_fully_connected_layers==0:  # If there's no fully connected layers, just sum up the elementwise multiplication (dot product)
        last_layer = tf.reduce_mean(elementwise_multiplied, axis=1)
    else:  # Otherwise, add the fc layers as needed
        net = slim.repeat(elementwise_multiplied, num_fully_connected_layers, slim.fully_connected, 100, scope='last_fc_layers')
        last_layer = slim.fully_connected(net, 1, scope='final_fc')



    print("Final shape: ")
    print(last_layer.get_shape())

    logits = last_layer
    return logits
