#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

import overlap_input
from constants import FLAGS
from vae import TangoEncoder

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


class TrainingException(Exception):
    pass


def train(sess, batch_size=FLAGS.BATCH_SIZE, training_epochs=60, display_step=5):
    vae = TangoEncoder(sess=sess)

    try:
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_samples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                # batch_xs, _ = mnist.train.next_batch(batch_size)
                batch_xs = images_batch.eval()
                overlap_areas = labels_batch.eval()

                # Fit training using batch data
                # cost = vae.partial_fit(batch_xs)
                cost, training_loss, rec_loss, lat_loss, def_loss = vae.partial_fit(batch_xs, overlap_areas=overlap_areas)

                if np.isnan(cost):
                    raise TrainingException("Got cost=nan")

                fill = ' '

                # The training_loss is the actual loss for calculating the overlap area (i.e. learning the metric space)
                # All other losses are for the autoencoder to actually create a reasonable latent space in the first place!
                print(("Epoch: (" + str(epoch) + "/" + str(training_epochs) + ");").ljust(16, fill) + (" i: (" + str(i) + "/" + str(
                    total_batch) + ").").ljust(16, fill)),
                print(("TRAINING LOSS: " + str(float(training_loss))).ljust(35, fill)),
                print(("Current cost: " + str(cost) + "").ljust(30, fill)),
                print(("[Reconstruction loss: " + str(rec_loss) + ";").ljust(32, fill) + (" latent loss: " + str(
                    lat_loss) + ";").ljust(25, fill) + ("deformation loss: " + str(def_loss) + ".").ljust(27, fill) + "]")


                if i % 24 == 0:
                    send_cross_entropy_to_poor_mans_tensorboard(training_loss)


                # Compute average loss
                avg_cost += cost / n_samples * batch_size

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1),
                      "cost=", "{:.9f}".format(avg_cost))

    except KeyboardInterrupt:
        pass

    return vae



with tf.Session() as sess:

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    try:
        # Train the autoencoder for 75 epochs
        vae = train(sess=sess, training_epochs=75)

        # Initialize all variables
        # sess.run(tf.global_variables_initializer())


        print("Reconstructing test input..."),

        # Display the input reconstruction
        x_sample = images_batch.eval()
        overlap_areas = labels_batch.eval()
        x_sample = np.reshape(x_sample, newshape=[-1, FLAGS.IMAGE_SIZE * FLAGS.IMAGE_SIZE * FLAGS.NUM_LAYERS])
        x_reconstruct = vae.reconstruct(x_sample, overlap_areas=overlap_areas)

        plt.figure(figsize=(8, 12))
        for i in range(5):
            # print("x_sample[i] shape: "),
            # print(np.shape(x_sample[i]))
            # print("")
            # print("x_reconstruct[i] shape: ")
            # print(np.shape(x_reconstruct[i]))
            # print("")

            plt.subplot(5, 2, 2 * i + 1)
            plt.imshow(x_sample[i].reshape(200, 200, FLAGS.NUM_LAYERS)[:, :, 0], vmin=0, vmax=1, cmap="gray")
            # plt.imshow(x_sample[i].reshape(200, 200, 1)[:, :, 1], vmin=0, vmax=1, cmap="gray")
            plt.title("Test input")
            plt.colorbar()
            plt.subplot(5, 2, 2 * i + 2)
            # plt.imshow(x_reconstruct[i].reshape(200, 200, FLAGS.NUM_LAYERS)[:, :, 1], vmin=0, vmax=1, cmap="gray")
            plt.imshow(x_reconstruct[i].reshape(200, 200, 1)[:, :, 0], vmin=0, vmax=1, cmap="gray")
            plt.title("Reconstruction")
            plt.colorbar()
        plt.tight_layout()

        plt.savefig('foo.png')

        print("Done!")

        print("Sampling 2d latent space..."),

        if vae.latent_dimensions == 2:
            nx = ny = 20
            x_values = np.linspace(-3, 3, nx)
            y_values = np.linspace(-3, 3, ny)

            canvas = np.empty((200 * ny, 200 * nx))
            for i, yi in enumerate(x_values):
                for j, xi in enumerate(y_values):
                    z_mu = np.array([[xi, yi]] * vae.batch_size)
                    x_mean = vae.generate(z_mu)
                    # canvas[(nx - i - 1) * 200:(nx - i) * 200, j * 200:(j + 1) * 200] = x_mean[0].reshape(200, 200, FLAGS.NUM_LAYERS)[:, :,
                    #                                                                    0]
                    canvas[(nx - i - 1) * 200:(nx - i) * 200, j * 200:(j + 1) * 200] = x_mean[0].reshape(200, 200, 1)[:, :,
                                                                                       0]

            plt.figure(figsize=(8, 10))
            Xi, Yi = np.meshgrid(x_values, y_values)
            plt.imshow(canvas, origin="upper", cmap="gray")
            plt.tight_layout()

            plt.savefig('latent_space_2d_sampling.png')

            print("Done!")
        else:
            print("Number of latent dimensions≠2")

    except KeyboardInterrupt:
        print("Good-by!")

        sys.exit(0)
