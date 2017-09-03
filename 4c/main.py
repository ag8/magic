from __future__ import print_function

import sys
import os

import datetime
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

import overlap_input
from constants import FLAGS
from utils import *
from vae import VariationalAutoencoder

# Load MNIST data in a format suited for tensorflow.
# The script input_data is available under this URL:
# https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/input_data.py
# import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# n_samples = mnist.train.num_examples

print("Running experiment " + os.path.basename(os.path.dirname(os.path.abspath(__file__))) + "!")


# Get input data
images_batch, labels_batch = overlap_input.inputs(reshape=True)
n_samples = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN


def train(network_architecture, sess, learning_rate=0.001,
          batch_size=FLAGS.BATCH_SIZE, training_epochs=10,
          step_display_step=5, epoch_display_step=5):

    vae = VariationalAutoencoder(network_architecture, sess=sess,
                                 transfer_fct=tf.nn.tanh,  # FIXME: Fix numerical issues instead of just using tanh
                                 learning_rate=learning_rate,
                                 batch_size=batch_size)
    try:
        loop_start = datetime.datetime.now()

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_samples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                # batch_xs, _ = mnist.train.next_batch(batch_size)
                batch_xs = images_batch.eval()

                # Fit training using batch data
                cost, r_loss_op, l_loss_op = vae.partial_fit(batch_xs)
                vae.save()

                if i % step_display_step == 0:
                    loop_end = datetime.datetime.now()
                    diff = (loop_end - loop_start).total_seconds()
                    exps = float(step_display_step) / diff

                    print("Epoch: (" + str(epoch) + "/" + str(training_epochs) + "); i: (" + str(i) + "/" + str(total_batch) + ").  Current cost: " + str(cost) + "           (exps: " + ("%.2f" % exps) + ")")

                    loop_start = datetime.datetime.now()

                if np.isnan(cost):
                    print("Cost is nan!! Killing everything.")
                    sys.exit(1)

                # Compute average loss
                avg_cost += cost / n_samples * batch_size

            # Display logs per epoch step
            if epoch % epoch_display_step == 0:
                print("Epoch:", '%04d' % (epoch+1),
                      "cost=", "{:.9f}".format(avg_cost))

    except KeyboardInterrupt:
        print("ancelling--Stopping training")
        return vae


    return vae




with tf.Session() as sess:
    printl("Defining network architecture...")

    # Define the network architecture
    network_architecture = \
        dict(num_neurons_recognition_layer_1=500,  # 1st layer encoder neurons
             num_neurons_recognition_layer_2=500,  # 2nd layer encoder neurons
             num_neurons_generator_layer_1=500,  # 1st layer decoder neurons
             num_neurons_generator_layer_2=500,  # 2nd layer decoder neurons
             num_input_neurons=FLAGS.IMAGE_SIZE * FLAGS.IMAGE_SIZE * 2,  # MNIST data input (img shape: 28*28)
             n_z=2)  # dimensionality of latent space

    print("Done.")


    printl("Launching filename queue population...")

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    print("Done.")


    try:
        print("Training autoencoder...")
        # Train the autoencoder for 75 epochs
        vae = train(network_architecture, sess=sess, training_epochs=20)
        print("Done")

        # Initialize all variables
        # sess.run(tf.global_variables_initializer())


        printl("Reconstructing test input...")

        # Display the input reconstruction
        x_sample = images_batch.eval()
        x_sample = np.reshape(x_sample, newshape=[-1, FLAGS.IMAGE_SIZE * FLAGS.IMAGE_SIZE * 2])
        x_reconstruct = vae.reconstruct(x_sample)

        plt.figure(figsize=(8, 12))
        for i in range(5):
            # print("x_sample[i] shape: "),
            # print(np.shape(x_sample[i]))
            # print("")
            # print("x_reconstruct[i] shape: ")
            # print(np.shape(x_reconstruct[i]))
            # print("")

            plt.subplot(5, 2, 2 * i + 1)
            plt.imshow(x_sample[i].reshape(200, 200, 2)[:, :, 0], vmin=0, vmax=1, cmap="gray")
            plt.title("Test input")
            plt.colorbar()
            plt.subplot(5, 2, 2 * i + 2)
            plt.imshow(x_reconstruct[i].reshape(200, 200, 2)[:, :, 0], vmin=0, vmax=1, cmap="gray")
            plt.title("Reconstruction")
            plt.colorbar()
        plt.tight_layout()

        plt.savefig('reconstruction_quality.png')

        print("Done!")


        # printl("Drawing 2d latent space representation...")  # TODO: Only if 2d

        # x_sample, y_sample = images_batch.eval(), labels_batch.eval()
        # z_mu = vae.transform(x_sample)
        # plt.figure(figsize=(8, 6))
        # plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
        # plt.colorbar()
        # plt.grid()

        # plt.savefig('latent_space_2d_plot.png')

        # print("Done!")


        printl("Sampling 2d latent space...")  # TODO: Only if 2d

        nx = ny = 20
        x_values = np.linspace(-3, 3, nx)
        y_values = np.linspace(-3, 3, ny)

        canvas = np.empty((200 * ny, 200 * nx))
        for i, yi in enumerate(x_values):
            for j, xi in enumerate(y_values):
                z_mu = np.array([[xi, yi]] * vae.batch_size)
                x_mean = vae.generate(z_mu)
                canvas[(nx - i - 1) * 200:(nx - i) * 200, j * 200:(j + 1) * 200] = x_mean[0].reshape(200, 200, 2)[:, :, 0]

        plt.figure(figsize=(8, 10))
        Xi, Yi = np.meshgrid(x_values, y_values)
        plt.imshow(canvas, origin="upper", cmap="gray")
        plt.tight_layout()

        plt.savefig('latent_space_2d_sampling.png')

        print("Done!")


    except KeyboardInterrupt:
        print("ome again! Good-by!")

        sys.exit(0)

