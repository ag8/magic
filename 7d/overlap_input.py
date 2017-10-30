#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import pickle
import sys

from six.moves import xrange

from constants import FLAGS


sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))
# import affinity as af
from affinity.models.magic_autoencoder.utils.utils import *


def read_images(filename_queue):
    """
    Reads a pair of lock/key images, and the overlap area, based on the filename queue.

    :param filename_queue: the filename queue of lock/key files.
    :return: a triple containing the lock image, the key image,
             and their maximum overlap area, in pixels.
    """
    # Each element in the queue contains three items:
    # the lock image filename, the key image filename,
    # and the maximum overlap area between the lock
    # and the key, in pixels.
    _, lock_image = decode_mshapes(filename_queue[0])
    _, key_image = decode_mshapes(filename_queue[1])
    overlap_area = filename_queue[2]

    # Return the examples
    return lock_image, key_image, overlap_area


def decode_mshapes(file_path):
    """
    Decodes an MSHAPE record.

    :param file_path: The filepath of the png
    :return: A duple containing 0 and the decoded image tensor
             (the zero is for compatibility reasons)
    """

    # Read the whole file
    serialized_record = tf.read_file(file_path)

    # Decode everything into uint8
    image = tf.image.decode_png(serialized_record, dtype=tf.uint8)

    # Cast to float32
    image = tf.cast(image, tf.float32)

    # "Reshape" the image to its expected size.
    # This does not actually do anything, since
    # the image remains the same size; however,
    # it has the effect of setting the tensor
    # shape so that it is inferred correctly
    # in later steps. For details, please
    # see https://stackoverflow.com/a/35692452
    # (however, instead of abusing random_crop,
    #  we simply reshape the image into itself)
    image = tf.reshape(image, [FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, 1])

    # Return zero and the image
    # (the zero is for compatibility
    #  with the JSHAPES input pipeline)
    return 0, image


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

    # First, check whether a data
    # directory has been supplied
    if not FLAGS.DATA_DIR:
        raise ValueError('Please supply a data_dir')

    # Get the images and the labels
    if not rotation:
        (images, labels) = _inputs(data_dir=FLAGS.DATA_DIR,
                                   batch_size=FLAGS.BATCH_SIZE)
    else:
        (images, labels) = _inputs_with_rotation(data_dir=FLAGS.DATA_DIR,
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


def _inputs_with_rotation(data_dir, batch_size):
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
    with open('/data/affinity/2d/overlap/OVERLAP_AREAS') as fp:
        overlap_areas = pickle.load(fp)
        overlap_areas = overlap_areas[:FLAGS.NUM_EXAMPLES_TO_LOAD_INTO_QUEUE]

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


    # Create a queue containing the lock filenames,
    # the key filenames, and the maximum overlap
    # area between the lock and the key.
    pairs_and_overlap_queue = tf.train.slice_input_producer([lock_files, key_files, overlap_areas],
                                                            shuffle=True)

    # print("Finished enqueueing")

    # Read the lock and key images from their filenames,
    # and get their maximum overlap area.
    lock_image, key_image, overlap_area = read_images(pairs_and_overlap_queue)

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
    print("Mixing properties stuff")
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)

    # Generate an image and label batch, and return it.
    return _generate_image_and_label_batch(image=combined_example, label=overlap_area,
                                           min_queue_examples=min_queue_examples,
                                           batch_size=batch_size,
                                           shuffle=True)


def _inputs(data_dir, batch_size):
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
    with open('/data/affinity/2d/overlap/OVERLAP_AREAS') as fp:
        overlap_areas = pickle.load(fp)
        overlap_areas = overlap_areas[:FLAGS.NUM_EXAMPLES_TO_LOAD_INTO_QUEUE]

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


    # Create a queue containing the lock filenames,
    # the key filenames, and the maximum overlap
    # area between the lock and the key.
    pairs_and_overlap_queue = tf.train.slice_input_producer([lock_files, key_files, overlap_areas],
                                                            shuffle=True)

    # print("Finished enqueueing")

    # Read the lock and key images from their filenames,
    # and get their maximum overlap area.
    lock_image, key_image, overlap_area = read_images(pairs_and_overlap_queue)

    # print("l shape: ", l.get_shape())
    # print("k shape: ", k.get_shape())

    # Combine the lock and key images into one image
    # with shape [IMAGE_SIZE, IMAGE_SIZE, 2].
    combined_example = tf.concat([lock_image, key_image], axis=2)

    # print("Combined example shape:--------------------------------------------------->", combined_example.get_shape())

    print("Got examples")

    # Ensure that the random shuffling has good mixing properties.
    print("Mixing properties stuff")
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)

    # Generate an image and label batch, and return it.
    return _generate_image_and_label_batch(image=combined_example, label=overlap_area,
                                           min_queue_examples=min_queue_examples,
                                           batch_size=batch_size,
                                           shuffle=True)


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.
    Args:
      image: 3-D Tensor of [height, width, 2] of type.float32.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 6] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    # print("aImages dimensions: ", image.get_shape())
    # print("aLabels dimensions: ", label.get_shape())

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

    # print("bImages dimensions: ", images.get_shape())
    # print("bLabels dimensions: ", label_batch.get_shape())

    # Return the images and the labels batches.
    return images, tf.reshape(label_batch, [batch_size])
