import pickle
from random import randint

import numpy as np
import scipy.misc

import tensorflow as tf

import overlap_input

from constants import FLAGS


# Get examples
images_batch, labels_batch = overlap_input.inputs()

def brilliant_neural_network(images, labels):


    # logits = tf.random_uniform(shape=[], minval=0, maxval=10000, dtype=tf.int32)

    logits = tf.constant(0)

    return logits

# Beautiful neural network
logits = brilliant_neural_network(images_batch, labels_batch)


# Loss function
def get_loss(logits, labels):
    logits = tf.cast(logits, dtype=tf.float32)
    labels = tf.cast(labels, dtype=tf.float32)
    return tf.reduce_mean(tf.squared_difference(logits, labels))


loss = get_loss(logits=logits, labels=labels_batch)

# Optimizer
# train_op = tf.train.AdamOptimizer(FLAGS.LEARNING_RATE).minimize(loss)



with open('OVERLAP_AREAS') as fp:
    overlap_areas = pickle.load(fp)
    overlap_areas = overlap_areas[:FLAGS.NUM_EXAMPLES_TO_LOAD_INTO_QUEUE]

print(overlap_areas)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    loss_sum = sess.run(loss)
    for i in range(0, FLAGS.MAX_TRAIN_ITERATIONS):
        # images, labels = sess.run([images_batch, labels_batch])
        # first_lock = images[0, :, :, 0]
        # first_key = images[0, :, :, 1]
        # print(np.shape(first_lock))
        # scipy.misc.imsave('out_L.png', first_lock)
        # scipy.misc.imsave('out_K.png', first_key)
        #
        # print("Label: ", labels[0])

        # _, my_loss = sess.run([train_op, loss])
        my_loss = sess.run(loss)

        loss_sum = loss_sum + my_loss
        avg_loss = loss_sum / i

        print("Step: " + str(i) + ", avg loss: " + str(avg_loss) + ".")
