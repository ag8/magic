#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The idea for this input pipe is as follows:

We create a queue where we get five images at a time.

The images are:

    - [0] Key A
    - [1] Lock A, rotation 1
    - [2] Lock A, rotation 2
    - [3] Lock B, rotation 1
    - [4] Lock B, rotation 2

Then, we return a queue of all five images (we don't choose randomly).

Then, inside the autoencoder, we take a correct pair ([0] and [1]),
and an incorrect pair ([0] and [3]), and use those as a training example.

Then, we encode ([0] and [2]), and we give it a penalty if it's too
dissimilar from ([0] and [1]). We also encode ([0] and [4]), and give
it a penalty if it's too dissimilar from ([0] and [3]). Finally,
we give a deformation penalty if ([0], [1/2]) and ([0], [3/4]) are similar.

However, the input pipe doesn't do any of the loss logic--all the input pipe
needs to do is take in images and rotate some of them.
"""


# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
from constants import FLAGS


import sys
from random import randint

import numpy as np

from six.moves import xrange
import tensorflow as tf


sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))
# import affinity as af
from affinity.models.magic_autoencoder.utils.utils import *


def read_mshapes_correct(filename_queue):
    """
    Reads a pair of MSHAPE records from the filename queue.

    :param filename_queue: the filename queue of lock/key files.
    :return: a duple containing a correct example and an incorrect example
    """
    _, lock_image = decode_mshapes(filename_queue[0])
    _, key_image = decode_mshapes(filename_queue[1])

    # Combine images to make a correct example and an incorrect example
    # correct_example = tf.concat([lock_image, key_image], axis=0)

    # print("Correct example", correct_example)

    # Return the examples
    return lock_image, key_image


def read_mshapes_incorrect(filename_queue):
    """
    Reads a pair of MSHAPE records from the filename queue.

    :param filename_queue: the filename queue of lock/key files.
    :return: a duple containing a correct example and an incorrect example
    """
    _, wrong_key_image = decode_mshapes(filename_queue[0])

    return wrong_key_image


def decode_mshapes(file_path):
    """
    Decodes an MSHAPE record.

    :param file_path: The filepath of the png
    :return: A duple containing 0 and the decoded image tensor
    """

    # read the whole file
    serialized_record = tf.read_file(file_path)

    # decode everything into uint8
    image = tf.image.decode_png(serialized_record, dtype=tf.uint8)

    # Cast to float32
    image = tf.cast(image, tf.float32)

    # "Crop" the image.
    # This does not actually do anything, since
    # the image remains the same size; however,
    # it has the effect of setting the tensor
    # shape so that it is inferred correctly
    # in later steps. For details, please
    # see https://stackoverflow.com/a/35692452
    # image = tf.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    image = tf.reshape(image, [FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, 1])

    return 0, image


def inputs(eval_data):
    """Construct input for MSHAPES evaluation using the Reader ops.
    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
    :returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.DATA_DIR:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.DATA_DIR, '')
    (filequeue, (images, labels)) = _inputs(eval_data=eval_data,
                                                    data_dir=data_dir,
                                                    batch_size=FLAGS.BATCH_SIZE)

    # print("Reenqueues: ")
    # print(reenqueues)

    if FLAGS.USE_FP16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)

    return filequeue, images, labels


def _inputs(eval_data, data_dir, batch_size):
    """
    Constructs the input for MSHAPES.

    :param eval_data: boolean, indicating if we should use the training or the evaluation data set
    :param data_dir: Path to the MSHAPES data directory
    :param batch_size: Number of images per batch

    :return:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 6] size
        labels: Labels. 1D tensor of [batch_size] size.
    """
    lock_files = []
    key_files_good = []
    key_files_bad = []

    if not eval_data:
        print("Not eval data")

        print_progress_bar(FLAGS.MIN_FILE_NUM, FLAGS.NUM_EXAMPLES_TO_LOAD_INTO_QUEUE, prefix='Progress:', suffix='Complete', length=50,
                           fill='█')

        for i in xrange(FLAGS.MIN_FILE_NUM, FLAGS.NUM_EXAMPLES_TO_LOAD_INTO_QUEUE, 2):
            print_progress_bar(i + 1, FLAGS.NUM_EXAMPLES_TO_LOAD_INTO_QUEUE, prefix='Progress:', suffix='Complete',
                               length=50,
                               fill='█')

            lock = os.path.join(data_dir, '%d_L.png' % i)
            key_good = os.path.join(data_dir, '%d_K.png' % i)
            key_bad = os.path.join(data_dir, '%d_K.png' % (i + 1))

            lock_files.append(lock)
            key_files_good.append(key_good)
            key_files_bad.append(key_bad)

        num_examples_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        print("Ok")
    else:
        die("Please use separate eval function!")

        num_examples_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    # print("Lock files: ")
    # print(lock_files)
    # print("Good key files: ")
    # print(key_files_good)
    # print("Bad key files: ")
    # print(key_files_bad)

    good_pairs_queue = tf.train.slice_input_producer([lock_files, key_files_good],
                                                     num_epochs=None, shuffle=True)
    bad_pairs_queue = tf.train.slice_input_producer([key_files_bad],
                                                    num_epochs=None, shuffle=True)

    print("Finished enqueueing")

    # Get the correct and incorrect examples from files in the filename queue.
    l, k = read_mshapes_correct(good_pairs_queue)
    wk = read_mshapes_incorrect(bad_pairs_queue)

    correct_example = tf.concat([l, k], axis=2)
    wrong_example = tf.concat([l, wk], axis=2)

    print("c Example shape:--------------------------------------------------------->", correct_example.get_shape())
    print("w Example shape:--------------------------------------------------------->", wrong_example.get_shape())

    print("Got examples")

    # Ensure that the random shuffling has good mixing properties.
    print("Mixing properties stuff")
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Regroup the enqueues
    # grouped_enqueues = tf.group(enqueues[0], enqueues[1])
    # for i in xrange(2, len(enqueues) - 1):
    #     grouped_enqueues = tf.group(grouped_enqueues, enqueues[i])

    correct_or_incorrect = tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)

    # The case code is basically tensorflow language for this:
    #
    # if (correct_or_incorrect < 0.5):
    #     _generate_image_and_label_batch(correct_example, [1],
    #                                     min_queue_examples, batch_size,
    #                                     shuffle=False)
    # else:
    #     _generate_image_and_label_batch(wrong_example, [0],
    #                                     min_queue_examples, batch_size,
    #                                     shuffle=False)

    def f1():
        return correct_example

    def f2():
        return wrong_example

    def g1():
        return tf.constant(0)

    def g2():
        return tf.constant(1)

    image = tf.case(
        {tf.less(correct_or_incorrect, tf.constant(0.5)): f1, tf.greater(correct_or_incorrect, tf.constant(0.5)): f2},
        default=f1, exclusive=True)
    # image = tf.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 6])
    image = tf.reshape(image, [FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, 2])
    label = tf.case(
        {tf.less(correct_or_incorrect, tf.constant(0.5)): g1, tf.greater(correct_or_incorrect, tf.constant(0.5)): g2},
        default=g1, exclusive=True)

    return (good_pairs_queue, (_generate_image_and_label_batch(image, label,
                                                               min_queue_examples, batch_size,
                                                               shuffle=True)))

    # def f1(): return (good_pairs_queue, (_generate_image_and_label_batch(correct_example, [1],
    #                                     min_queue_examples, batch_size,
    #                                     shuffle=False)))
    # def f2(): return (good_pairs_queue, (_generate_image_and_label_batch(wrong_example, [0],
    #                                     min_queue_examples, batch_size,
    #                                     shuffle=False)))
    #
    # r = tf.case([(tf.less(correct_or_incorrect, tf.constant(0.5)), f1)], default=f2)
    #
    # return r


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.
    Args:
      image: 3-D Tensor of [height, width, 6] of type.float32.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 6] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    print("Image dimensions: ", image.get_shape())
    # image = tf.reshape(image, [2 * IMAGE_SIZE, IMAGE_SIZE, 3])

    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = FLAGS.NUM_THREADS
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 6 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        print("Not shuffling!")
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 6 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    print("Images dimensions: ", images.get_shape())

    return images, tf.reshape(label_batch, [batch_size])
