#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
import scipy.misc

from six.moves import xrange

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from constants import FLAGS

import overlap_input


sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))
# import affinity as af
from utils import *


maxareas = []
with open(FLAGS.DATA_DIR + '/OVERLAP_AREAS') as fp:
    maxareas = pickle.load(fp)


def simple_inputs(data_dir, batch_size):
    """
        Internally constructs the input for overlap dataset.

        :param data_dir: Path to the overlap data directory
        :param batch_size: Number of images per batch

        :return:
            images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 6] size
            labels: Labels. 1D tensor of [batch_size] size.
        """

    # First, we make lists of filenames for
    # the lock files and the key files
    lock_files = []
    key_files = []

    # Now, we read the array of overlap areas
    # from a pickled file. This file is generally
    # stored in the same directory as this file,
    # under the name `OVERLAP_AREAS`.
    overlap_areas = []
    with open(FLAGS.DATA_DIR + '/OVERLAP_AREAS') as fp:
        overlap_areas = pickle.load(fp)
        overlap_areas = overlap_areas[:FLAGS.NUM_EXAMPLES_TO_LOAD_INTO_QUEUE]

    print("Overlap areas: " + str(overlap_areas))

    # Initialize the progress bar
    print_progress_bar(0, FLAGS.NUM_EXAMPLES_TO_LOAD_INTO_QUEUE, prefix='Progress:', suffix='Complete', length=50)

    # Load the requested number of lock/key image
    # filenames into their respective lists
    for i in xrange(0, FLAGS.NUM_EXAMPLES_TO_LOAD_INTO_QUEUE):
        # The lock and key files are always in the format
        # {N}_L.png and {N}_K.png, respectively, where {N}
        # is the id of the lock/key pair
        lock = os.path.join(data_dir, '%d_L.png' % i)
        key = os.path.join(data_dir, '%d_K.png' % i)

        # Append the filenames to their respective lists
        lock_files.append(lock)
        key_files.append(key)

        # Update the progress bar
        print_progress_bar(i + 1, FLAGS.NUM_EXAMPLES_TO_LOAD_INTO_QUEUE, prefix='Progress:', suffix='Complete',
                           length=50)

    # lock_files = list(reversed(lock_files))
    # key_files = list(reversed(key_files))

    print("Lock files: " + str(lock_files))

    # Create a queue containing the lock filenames,
    # the key filenames, and the maximum overlap
    # area between the lock and the key.
    pairs_and_overlap_queue = tf.train.slice_input_producer([lock_files, key_files, overlap_areas],
                                                            capacity=100000,
                                                            shuffle=True)

    # print("Finished enqueueing")

    # Read the lock and key images from their filenames,
    # and get their maximum overlap area.
    lock_image, key_image, overlap_area = overlap_input.read_images(pairs_and_overlap_queue)

    rotated_lock_image, _ = random_rotation(lock_image, image_size=FLAGS.IMAGE_SIZE)
    rotated_lock_image = tf.reshape(rotated_lock_image, shape=[FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, 1])

    # print("l shape: ", l.get_shape())
    # print("k shape: ", k.get_shape())

    # Combine the lock and key images into one image
    # with shape [IMAGE_SIZE, IMAGE_SIZE, 2].
    combined_example = tf.concat([lock_image, rotated_lock_image, key_image], axis=2)

    # print("Combined example shape:--------------------------------------------------->", combined_example.get_shape())

    print("Got examples")

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)

    # Generate an image and label batch, and return it.
    return overlap_input._generate_image_and_label_batch(image=combined_example, label=overlap_area,
                                           min_queue_examples=min_queue_examples,
                                           batch_size=batch_size,
                                           shuffle=True)


def inputs(normalize=False, reshape=False, rotation=False):
    """Constructs input for the overlap dataset

    :param normalize: Whether to normalize the pixel data to the range [0, 1).
                      By default, the pixel data is in the range [0, 255].
                      (Normalization is critical for architectures involving
                      variational autoencoders!)
    :param reshape: Whether to reshape the images into long vectors
    :param rotation: Whether to return rotated versions of the lock images
                     along with the normal lock images

    :returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, N] size,
              where N=2 if rotation=False, and N=3 if rotation=True.
      labels: Labels. 1D tensor of [batch_size] size.

    :raises:
      ValueError: If no data_dir
    """
    if not rotation:
        raise ValueError("Rotation has to be True.")

    # First, check whether a data
    # directory has been supplied
    if not FLAGS.DATA_DIR:
        raise ValueError('Please supply a data_dir')

    # Get the images and the labels
    # (images, labels) = _inputs_with_rotation(data_dir=FLAGS.DATA_DIR,
    #                                              batch_size=FLAGS.BATCH_SIZE)

    # Get the images and the labels
    (images, labels) = simple_inputs(data_dir=FLAGS.DATA_DIR,
                                             batch_size=FLAGS.BATCH_SIZE)

    # Optionally, cast the images and labels to float16
    # (by default, we use the float32 data type).
    if FLAGS.USE_FP16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)

    if normalize:
        # Normalize the pixel data to [0, 1), if necessary
        images = tf.divide(images, tf.constant(255.0))

    if reshape:
        # Reshape the images to a long vector, if necessary
        images = tf.reshape(images, shape=[FLAGS.BATCH_SIZE, FLAGS.IMAGE_SIZE * FLAGS.IMAGE_SIZE * FLAGS.NUM_LAYERS])

    # Return the images and the labels.
    return images, labels



# Get input data
images_batch, labels_batch = overlap_input.inputs(normalize=True, reshape=False, rotation=FLAGS.ROTATE)
n_samples = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN


with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for i in xrange(0, 20):
        print("Getting batch of images and labels.")

        current_image_batch = images_batch.eval()
        current_labels_batch = labels_batch.eval()

        print("labels batch: ")
        print(current_labels_batch)

        print("")
        print("")
        print("Consider the 2nd image.")

        first_image = current_image_batch[4][0, :, 0]
        print(np.shape(first_image))

        print("Overlap: " + str(current_labels_batch[4]))
        print("Index: " + str(maxareas.index(current_labels_batch[4])))
        scipy.misc.imsave("" + str(current_labels_batch[4]) + ".png", current_image_batch[0])
