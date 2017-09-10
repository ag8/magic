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

import overlap_input
from constants import FLAGS
from vae import TangoEncoder

from excps import *

from utils import *
import debugutils as dbu



# Get the raw overlap areas and generate hashes of the images
# for intermediate testing.
if FLAGS.RUN_INTERMEDIATE_TESTS:
    max_overlap = []
    with open('/data/affinity/2d/overlap_micro/OVERLAP_AREAS') as fp:
        max_overlap = pickle.load(fp)
        print(max_overlap)

    image_hashes = dbu.generate_lock_hashes(FLAGS.DATA_DIR)


# Get input data
images_batch, labels_batch = overlap_input.inputs(normalize=True, reshape=True, rotation=FLAGS.ROTATE)
n_samples = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN


def train(sess, batch_size=FLAGS.BATCH_SIZE, training_epochs=60):
    """
    Train an autoencoder, and evaluate the accuracy.

    :param sess: the tf session to run all operations in
    :param batch_size: the batch size for the examples
    :param training_epochs: the maximum number of epochs for training
    :return: the autoencoder network.
    """
    vae = TangoEncoder(sess=sess)

    try:
        # Training cycle
        for epoch in range(training_epochs):
            # print("Epoch: " + str(epoch))

            avg_cost = 0.
            avg_loss = 0.
            total_batch = int(n_samples / batch_size)

            # Loop over all batches
            for i in range(total_batch):
                # Get a batch of images and labels
                batch_xs, overlap_areas = sess.run([images_batch, labels_batch])

                # Fit training using batch data
                cost, training_loss, rec_loss, lat_loss, def_loss, distance = vae.partial_fit(batch_xs,
                                                                                    overlap_areas=overlap_areas)

                print(("Epoch: " + str(epoch)).ljust(20)),
                print(("Cost: " + str(cost)).ljust(20)),
                print(("Training loss: " + str(training_loss).ljust(30))),
                print(("Distance: " + str(np.mean(distance))).ljust(20))

                # If the cost is nan, stop training and raise an exception
                if np.isnan(cost):
                    raise TrainingException("Got cost=nan")

                # The training_loss is the actual loss for calculating the overlap area (i.e. learning the metric space)
                # All other losses are for the autoencoder to actually create a reasonable latent space in the first place!
                # print_losses(epoch=epoch,
                #              training_epochs=training_epochs,
                #              total_batch=total_batch,
                #              i=i,
                #              training_loss=training_loss,
                #              cost=cost,
                #              rec_loss=rec_loss,
                #              lat_loss=lat_loss,
                #              def_loss=def_loss)

                # Every 100 steps, evaluate the accuracy
                if epoch % 10 == 0 and i == 3:
                    accuracy_testing_images, accuracy_testing_overlap_areas = sess.run([images_batch, labels_batch])
                    predictions = vae.get_predictions(accuracy_testing_images,
                                                      overlap_areas=accuracy_testing_overlap_areas)


                    # Invert the predictions--since the distance is the inverse
                    # predictions = 1. / predictions


                    print("Predictions: " + str(predictions))
                    print("Actual areas: " + str(accuracy_testing_overlap_areas))

                    mse = (np.array(predictions - accuracy_testing_overlap_areas) ** 2).mean(axis=None)

                    print("mse: " + str(mse) + "     "),

                    # save an example
                    save_example(accuracy_testing_images, accuracy_testing_overlap_areas,
                                 predictions,
                                 epoch, i, image_hashes, max_overlap)

                    # Send the cross entropy to the logger
                    send_cross_entropy_to_poor_mans_tensorboard(mse)

                # Compute average loss
                avg_cost += cost / n_samples * batch_size
                avg_loss += training_loss / n_samples * batch_size

            print(("Average loss per epoch: " + str(int(avg_loss)) + "").ljust(35, ' '))

    except KeyboardInterrupt:
        pass

    return vae



with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    try:
        # Train the autoencoder for 75 epochs
        vae = train(sess=sess, training_epochs=3000)

        print("Reconstructing test input..."),

        x_sample, overlap_areas = sess.run([images_batch, labels_batch])
        reconstruct_input(x_sample, overlap_areas, vae)

        print("Done!")

        print("Sampling 2d latent space..."),

        sample_latent_space(vae)

        print("Done!")

    except KeyboardInterrupt:
        print("Good-by!")

        sys.exit(0)
