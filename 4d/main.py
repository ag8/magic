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


LATENT_DIMENSIONS = 20


print("Running experiment " + os.path.basename(os.path.dirname(os.path.abspath(__file__))) + "!")


# Get input data from input pipe
images_batch, labels_batch = overlap_input.inputs(reshape=True)
n_samples = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN


def train(network_architecture, sess, learning_rate=FLAGS.LEARNING_RATE,
          batch_size=FLAGS.BATCH_SIZE, training_epochs=FLAGS.MAX_EPOCHS,
          step_display_step=5, epoch_display_step=5):
    """
    The training function builds the autoencoder and trains it.

    :param network_architecture: a set of variables corresponding to values defining
                                 the network architecture. See documentation for details.
    :param sess: the tensorflow session.
    :param learning_rate: the learning rate for the optimizer.
                          Defaults to value stored in FLAGS.
    :param batch_size: size of the images batch.
                    Defaults to value stored in FLAGS.
    :param training_epochs: Number of epochs to train.
                            Defaults to value stored in FLAGS.
    :param step_display_step: Display training info every {N} steps, default is 5.
    :param epoch_display_step: Display training info every {N} epochs, default is 5.
    :return:
    """


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

                if i % step_display_step == 0:
                    loop_end = datetime.datetime.now()
                    diff = (loop_end - loop_start).total_seconds()
                    exps = float(step_display_step) / diff

                    magic_print("Epoch: (" + str(epoch) + "/" + str(training_epochs) + "); i: (" + str(i) + "/" + str(total_batch) + ").  Current cost: " + str(cost) + "           (exps: " + ("%.2f" % exps) + ")", epoch=epoch, max_epochs=training_epochs, i=i, max_i=total_batch)

                    loop_start = datetime.datetime.now()

                if np.isnan(cost):
                    print("Cost is nan!! Killing everything.")
                    sys.exit(1)
                else:
                    # Only save the state of the autoencoder if the cost is valid.
                    # (This is needed, because if we save an autoencoder with
                    #  a cost of nan, we can't really load its weights, because
                    #  they're basically useless).
                    vae.save()

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
             n_z=LATENT_DIMENSIONS)  # dimensionality of latent space

    print("Done.")


    printl("Launching filename queue population...")

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    print("Done.")


    try:
        print("Training autoencoder...")
        # Train the autoencoder for 75 epochs
        vae = train(network_architecture, sess=sess, training_epochs=75)
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


        if LATENT_DIMENSIONS == 2:
            printl("Sampling 2d latent space...")

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

