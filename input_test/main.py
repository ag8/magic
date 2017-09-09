#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import os

import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import scipy.misc
import time

import overlap_input
from constants import FLAGS

from utils import *

# Load MNIST data in a format suited for tensorflow.
# The script input_data is available under this URL:
# https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/input_data.py
# import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# n_samples = mnist.train.num_examples

# Get input data
images_batch, labels_batch = overlap_input.inputs(normalize=True, reshape=True, rotation=FLAGS.ROTATE)
n_samples = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN


with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for i in xrange(0, 1):
        print("Getting batch of images and labels.")

        current_image_batch = images_batch.eval()
        current_labels_batch = labels_batch.eval()

        print("labels batch: ")
        print(current_labels_batch)
