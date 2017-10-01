#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import pickle

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import scipy.misc
import time

from constants import FLAGS
from vae import TangoEncoder

from excps import *

from utils import *
import debugutils as dbu

import PARAMS
import st5input



images_batch, labels_batch = st5input.inputs(eval_data=False, data_dir=PARAMS.DATA_DIR, batch_size=FLAGS.BATCH_SIZE)

# lock_images, lock_images_r, key_images = split(images_batch, num_splits=3)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for i in xrange(1, 2):

        # my_lock_images, my_lock_images_r, my_key_images, labels = sess.run([lock_images, lock_images_r, key_images, labels_batch])
        x_images, labels = sess.run([images_batch, labels_batch])

        print(np.shape(x_images))
        print(np.shape(labels))

        print(np.sum(x_images[0]))
        print(x_images[0])

        # print(np.shape(my_lock_images))
        # print(np.shape(my_lock_images_r))
        # print(np.shape(my_key_images))
        # print(labels)

        # print(np.sum(my_lock_images[0]))
