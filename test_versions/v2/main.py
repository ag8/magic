# from __future__ import print_function

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# from magic_input import *
import magic_input
from constants import FLAGS
import utils

# Load MNIST data in a format suited for tensorflow.
import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples


def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


class VariationalAutoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.

    This implementation uses probabilistic encoders and decoders using Gaussian
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.

    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """

    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus,
                 learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])

        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and
        # corresponding optimizer
        self._create_loss_optimizer()

        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def _create_network(self):
        # Initialize autoencode network weights and biases
        network_weights = self._initialize_weights(**self.network_architecture)

        # Use recognition network to determine mean and
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(network_weights["weights_recog"],
                                      network_weights["biases_recog"])

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1,
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean,
                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = \
            self._generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"])

    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2,
                            n_hidden_gener_1, n_hidden_gener_2,
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights

    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']),
                                           biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq = \
            tf.add(tf.matmul(layer_2, weights['out_log_sigma']),
                   biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']),
                                           biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        x_reconstr_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']),
                                 biases['out_mean']))
        return x_reconstr_mean

    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluation of log(0.0)
        reconstr_loss = \
            -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
                           + (1 - self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),
                           1)
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        ##    between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)  # average over batch
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def partial_fit(self, X):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost),
                                  feed_dict={self.x: X})
        return cost

    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})

    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.z: z_mu})

    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.x: X})


def train(network_architecture, images_batch, learning_rate=0.001, training_epochs=10, display_step=1):
    vae = VariationalAutoencoder(network_architecture,
                                 learning_rate=learning_rate,
                                 batch_size=FLAGS.BATCH_SIZE)

    # Initialize variables
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)




    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / FLAGS.BATCH_SIZE)
        total_batch = 4
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(FLAGS.BATCH_SIZE)

            # Reshape the images into one long tensor
            images_batch = tf.reshape(images_batch, shape=[-1, FLAGS.IMAGE_SIZE * FLAGS.IMAGE_SIZE * 2])
            images_batch = tf.divide(images_batch, tf.constant(255.0))

            # print("Current image batch: ")
            # print(batch_xs)
            # print("Current image batch shape: "),
            # print(np.shape(images_batch))

            # print("Images batch: "),
            # print(images_batch.eval())

            # Fit training using batch data
            # cost = vae.partial_fit(batch_xs)  # MNIST
            cost = vae.partial_fit(images_batch.eval())  # JSHAPES

            print("Epoch: (" + str(epoch) + "/ " + str(training_epochs) + "),  i: (" + str(i) + "/" + str(total_batch) + ").  Raw cost: " + str(cost))

            # Compute average loss
            avg_cost += cost / n_samples * FLAGS.BATCH_SIZE

        # Display logs per epoch step
        if epoch % display_step == 0:
            utils.parade_output("Epoch:" + str(epoch + 1) +
                  "  cost=" + "{:.9f}".format(avg_cost))
    return vae


# network_architecture = \
#     dict(n_hidden_recog_1=500, # 1st layer encoder neurons
#          n_hidden_recog_2=500, # 2nd layer encoder neurons
#          n_hidden_gener_1=500, # 1st layer decoder neurons
#          n_hidden_gener_2=500, # 2nd layer decoder neurons
#          n_input=784,  # MNIST
#          n_input=FLAGS.IMAGE_SIZE * FLAGS.IMAGE_SIZE * 2,  # JSHAPES
#          n_z=20)  # dimensionality of latent space


# Get examples
filequeue, images_batch, labels_batch = magic_input.inputs(eval_data=False)

# vae = train(network_architecture, images_batch=images_batch, training_epochs=2)

network_architecture = \
    dict(n_hidden_recog_1=500,  # 1st layer encoder neurons
         n_hidden_recog_2=500,  # 2nd layer encoder neurons
         n_hidden_gener_1=500,  # 1st layer decoder neurons
         n_hidden_gener_2=500,  # 2nd layer decoder neurons
         # n_input=784,  # MNIST
         n_input=FLAGS.IMAGE_SIZE * FLAGS.IMAGE_SIZE * 2,  # JSHAPES
         n_z=2)  # dimensionality of latent space


vae_2d = train(network_architecture, images_batch=images_batch, training_epochs=2)

nx = ny = 20
x_values = np.linspace(-3, 3, nx)
y_values = np.linspace(-3, 3, ny)

canvas = np.empty((200 * ny, 200 * nx))
for i, yi in enumerate(x_values):
    for j, xi in enumerate(y_values):
        z_mu = np.array([[xi, yi]] * vae_2d.batch_size)
        x_mean = vae_2d.generate(z_mu)
        x_mean = x_mean.reshape(FLAGS.BATCH_SIZE, 200, 200, 2)  # Yeah ok pretty sure this literally gives the mean
                                                                # (just looks like a 2d normal distribution, since
                                                                # random rotation/translation).
                                                                # I'll figure this out on Monday
        x_mean = x_mean[4, :, :, 0]
        print(x_mean)
        canvas[(nx - i - 1) * 200:(nx - i) * 200, j * 200:(j + 1) * 200] = x_mean

plt.figure(figsize=(8, 10))
Xi, Yi = np.meshgrid(x_values, y_values)
plt.imshow(canvas, origin="upper", cmap="gray")
plt.tight_layout()
plt.savefig('latent2d.png')


sys.exit(0)

# x_sample = mnist.test.next_batch(100)[0]  # MNIST
images_batch = tf.reshape(images_batch, shape=[-1, FLAGS.IMAGE_SIZE * FLAGS.IMAGE_SIZE * 2])  # FIXME: make a function out of this
images_batch = tf.divide(images_batch, tf.constant(255.0))
x_sample = images_batch.eval()  # JSHAPES
x_reconstruct = vae.reconstruct(x_sample)

plt.figure(figsize=(8, 12))
for i in range(5):
    combined_image = x_sample[i].reshape(200, 200, 2)

    lock_image = combined_image[:, :, 0]
    # print(lock_image)

    # print("Lock image shape: "),
    # print(np.shape(lock_image))


    reconstructed_combined_image = x_reconstruct[i].reshape(200, 200, 2)
    reconstructed_lock_image = reconstructed_combined_image[:, :, 0]


    plt.subplot(5, 2, 2*i + 1)
    plt.imshow(lock_image, vmin=0, vmax=1, cmap="gray")
    plt.title("Test input")
    plt.colorbar()
    plt.subplot(5, 2, 2*i + 2)
    plt.imshow(reconstructed_lock_image, vmin=0, vmax=1, cmap="gray")
    plt.title("Reconstruction")
    plt.colorbar()
plt.tight_layout()
plt.savefig('foo.png')
