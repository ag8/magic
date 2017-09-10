#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from constants import FLAGS


letters_data = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
cyrillic_letters_data = ["a", "b", "w", "g", "d", "e", "j", "v", "z", "i"]
numbers_data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]




def inputs(batch_size):
    # Get the letters and the labels
    (letters, labels) = _inputs(batch_size=batch_size)

    # Return the letters and the labels.
    return letters, labels


def read_letter(pairs_and_overlap_queue):
    # Read the letters, cyrillics, and numbers.
    letter = pairs_and_overlap_queue[0]
    cyrillic = pairs_and_overlap_queue[1]
    number = pairs_and_overlap_queue[2]

    # Do something with them
    # (doesn't matter what)
    letter = tf.substr(letter, 0, 1)
    cyrillic = tf.substr(cyrillic, 0, 1)
    number = tf.add(number, tf.constant(0))

    # Return them
    return letter, cyrillic, number


def _inputs(batch_size):
    # Get the input data
    letters = letters_data
    cyrillics = cyrillic_letters_data
    numbers = numbers_data


    # Create a queue containing the letters,
    # the cyrillics, and the numbers
    pairs_and_overlap_queue = tf.train.slice_input_producer([letters, cyrillics, numbers],
                                                            capacity=100000,
                                                            shuffle=True)

    # Perform some operations on each of those
    letter, cyrillic, number = read_letter(pairs_and_overlap_queue)

    # Combine the letters and cyrillics into one example
    combined_example = tf.stack([letter, cyrillic])


    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)

    # Generate an example and label batch, and return it.
    return _generate_image_and_label_batch(example=combined_example, label=number,
                                           min_queue_examples=min_queue_examples,
                                           batch_size=batch_size,
                                           shuffle=True)


def _generate_image_and_label_batch(example, label, min_queue_examples,
                                    batch_size, shuffle):

    # Create a queue that shuffles the examples, and then
    # read 'batch_size' examples + labels from the example queue.
    num_preprocess_threads = FLAGS.NUM_THREADS
    if shuffle:
        examples, label_batch = tf.train.shuffle_batch(
            [example, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 6 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        print("Not shuffling!")
        examples, label_batch = tf.train.batch(
            [example, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 6 * batch_size)

    # Return the examples and the labels batches.
    return examples, tf.reshape(label_batch, [batch_size])



lcs, nums = inputs(batch_size=3)



with tf.Session() as sess:

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)


    for i in xrange(0, 5):
        my_lcs, my_nums = sess.run([lcs, nums])

        print(str(my_lcs) + " --> " + str(my_nums))

