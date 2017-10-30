import sys
import os

import tensorflow as tf

from constants import FLAGS

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils import *


class TangoEncoder(object):
    def __init__(self, sess):
        self.sess = sess

        # Determine network architecture
        self.encoder_neurons = [500, 500]
        # self.encoder_neurons_layer1 = 500
        # self.encoder_neurons_layer2 = 500
        self.decoder_neurons_layer1 = 500
        self.decoder_neurons_layer2 = 500

        self.num_input_neurons = FLAGS.IMAGE_SIZE * FLAGS.IMAGE_SIZE * FLAGS.NUM_LAYERS

        self.latent_dimensions = 2

        self.transfer_fct = tf.nn.tanh
        self.learning_rate = 1e-4
        self.batch_size = FLAGS.BATCH_SIZE

        self.x = tf.placeholder(tf.float32, [None, self.num_input_neurons])
        print("x shape" + ("e" * 120)),
        print(self.x.get_shape())



        # Define the network architecture

        # Initialize encoder weights
        self.num_input_neurons_in_network = self.num_input_neurons / FLAGS.NUM_LAYERS

        num_encoder_layers = len(self.encoder_neurons)


        # START ENCODER WEIGHTS-----------------------------------------------------------------------------------------

        encoder_weights = []

        # Add zeroth layer
        print("Connecting network layer [" + str(self.num_input_neurons_in_network) + "->" + str(self.encoder_neurons[0]) + "].")
        encoder_h0_weights = tf.Variable(xavier_init(self.num_input_neurons_in_network, self.encoder_neurons[0]))
        encoder_weights.append(encoder_h0_weights)

        # Add as many layers as required
        for i in xrange(0, num_encoder_layers - 1):
            print("Connecting network layer [" + str(self.encoder_neurons[i]) + "->" + str(
                self.encoder_neurons[i+1]) + "].")
            encoder_hi_weights = tf.Variable(xavier_init(self.encoder_neurons[i], self.encoder_neurons[i+1]))
            encoder_weights.append(encoder_hi_weights)

        # Add the last layers
        encoder_out_mean_weights = tf.Variable(xavier_init(self.encoder_neurons[num_encoder_layers - 1], self.latent_dimensions))
        encoder_out_log_sigma_weights = tf.Variable(xavier_init(self.encoder_neurons[num_encoder_layers - 1], self.latent_dimensions))

        # END ENCODER WEIGHTS-------------------------------------------------------------------------------------------


        # START ENCODER BIASES------------------------------------------------------------------------------------------

        encoder_biases = []

        # Add zeroth layer
        print("Adding " + str(self.encoder_neurons[0]) + " biases.")
        encoder_b0_biases = tf.Variable(tf.zeros([self.encoder_neurons[0]], dtype=tf.float32))
        encoder_biases.append(encoder_b0_biases)

        for i in xrange(1, num_encoder_layers):
            print("Adding " + str(self.encoder_neurons[i]) + " biases.")
            encoder_bi_biases = tf.Variable(tf.zeros([self.encoder_neurons[i]], dtype=tf.float32))
            encoder_biases.append(encoder_bi_biases)

        print("Encoder biases length: "),
        print(len(encoder_biases))

        # Add the last layers
        encoder_out_mean_biases = tf.Variable(tf.zeros([self.latent_dimensions], dtype=tf.float32))
        encoder_out_log_sigma_biases = tf.Variable(tf.zeros([self.latent_dimensions], dtype=tf.float32))

        # END ENCODER BIASES--------------------------------------------------------------------------------------------

        decoder_h1_weights = tf.Variable(xavier_init(self.latent_dimensions, self.encoder_neurons[0]))
        decoder_h2_weights = tf.Variable(xavier_init(self.encoder_neurons[0], self.encoder_neurons[1]))
        decoder_out_mean_weights = tf.Variable(xavier_init(self.encoder_neurons[1], self.num_input_neurons_in_network))
        decoder_out_log_sigma_weights = tf.Variable(xavier_init(self.encoder_neurons[1], self.num_input_neurons_in_network))

        decoder_b1_biases = tf.Variable(tf.zeros([self.encoder_neurons[0]], dtype=tf.float32))
        decoder_b2_biases = tf.Variable(tf.zeros([self.encoder_neurons[1]], dtype=tf.float32))
        decoder_out_mean_biases = tf.Variable(tf.zeros([self.num_input_neurons_in_network], dtype=tf.float32))
        decoder_out_log_sigma_biases = tf.Variable(tf.zeros([self.num_input_neurons_in_network], dtype=tf.float32))

        self.reshaped_x = tf.reshape(self.x, [-1, FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, FLAGS.NUM_LAYERS])
        print("reshaped_x shape: "),
        print(self.reshaped_x.get_shape())
        lock_image, rotated_lock_image, key_image = split(self.reshaped_x, num_splits=3)
        lock_image = tf.reshape(lock_image, [-1, FLAGS.IMAGE_SIZE * FLAGS.IMAGE_SIZE])
        self.lock_image = lock_image
        rotated_lock_image = tf.reshape(rotated_lock_image, [-1, FLAGS.IMAGE_SIZE * FLAGS.IMAGE_SIZE])
        key_image = tf.reshape(key_image, [-1, FLAGS.IMAGE_SIZE * FLAGS.IMAGE_SIZE])

        self.z_mean, self.z_log_sigma_sq, self.z = self.get_latent_representation(lock_image,
                                                                                  encoder_weights,
                                                                                  encoder_biases,
                                                                                  encoder_out_mean_weights,
                                                                                  encoder_out_mean_biases,
                                                                                  encoder_out_log_sigma_weights,
                                                                                  encoder_out_log_sigma_biases)

        _, _, self.z_representation_of_rotated = self.get_latent_representation(rotated_lock_image,
                                                                                encoder_weights,
                                                                                encoder_biases,
                                                                                encoder_out_mean_weights,
                                                                                encoder_out_mean_biases,
                                                                                encoder_out_log_sigma_weights,
                                                                                encoder_out_log_sigma_biases)

        _, _, self.z_representation_of_key = self.get_latent_representation(key_image,
                                                                            encoder_weights,
                                                                            encoder_biases,
                                                                            encoder_out_mean_weights,
                                                                            encoder_out_mean_biases,
                                                                            encoder_out_log_sigma_weights,
                                                                            encoder_out_log_sigma_biases)







        # Decoder network
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, decoder_h1_weights),
                                           decoder_b1_biases))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, decoder_h2_weights),
                                           decoder_b2_biases))
        self.x_reconstructed_mean = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, decoder_out_mean_weights),
                                                    decoder_out_mean_biases))




        # Now, we create the loss optimizer
        with tf.name_scope("cost"):
            # The loss is composed of two terms:
            # 1.) The reconstruction loss (the negative log probability
            #     of the input under the reconstructed Bernoulli distribution
            #     induced by the decoder in the data space).
            #     This can be interpreted as the number of "nats" required
            #     for reconstructing the input when the activation in latent
            #     is given.
            # Adding 1e-10 to avoid evaluation of log(0.0)

            # in_reduce_sum = self.x * tf.log(1e-10 + self.x_reconstructed_mean) + (1 - self.x) * tf.log(1e-10 + 1 - self.x_reconstructed_mean)
            # in_reduce_sum = tf.log(tf.pow(self.x_reconstructed_mean, self.x)) + (1.0 - self.x) * tf.log(1.0 - self.x_reconstructed_mean)
            # in_reduce_sum = tf.log(tf.multiply(tf.pow(self.x_reconstructed_mean, self.x), tf.pow(1.0 - self.x_reconstructed_mean, 1.0 - self.x)))

            in_reduce_sum = tf.log(tf.multiply(tf.pow(self.x_reconstructed_mean, self.lock_image),
                                               tf.pow(1.0 - self.x_reconstructed_mean, 1.0 - self.lock_image)))

            reconstruction_loss = -tf.reduce_sum(in_reduce_sum, 1)
            self.r_l = tf.reduce_mean(reconstruction_loss)

            # 2.) The latent loss, which is defined as the Kullback-Leibler divergence
            ##    between the distribution in latent space induced by the encoder on
            #     the data and some prior. This acts as a kind of regularizer.
            #     This can be interpreted as the number of "nats" required
            #     for transmitting the the latent space distribution given
            #     the prior.   TODO: Try Jensen-Shannon divergence? (it's a metric and seems more stable)
            latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                               - tf.square(self.z_mean)
                                               - tf.exp(self.z_log_sigma_sq), 1)
            self.l_l = tf.reduce_mean(latent_loss)

            # 3.) The third part of the loss is the deformation loss;
            #     that is, the distance between two latent space
            #     representations of the lock image minus the distance
            #     between the representations of the lock image and
            #     the key image.
            deformation_loss = 10 * tf.reduce_sum(tf.squared_difference(self.z, self.z_representation_of_rotated), 1)
            self.d_l = tf.reduce_mean(deformation_loss)



            self.cost = tf.reduce_mean(reconstruction_loss + latent_loss + deformation_loss)  # average over batch


        with tf.name_scope("Adam_optimizer"):
            # Use ADAM optimizer
            self.optimizer = \
                tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)



        # Initialize all variables!
        self.sess.run(tf.global_variables_initializer())


    def get_latent_representation(self, X, encoder_weights, encoder_biases,
                                  encoder_out_mean_weights, encoder_out_mean_biases,
                                  encoder_out_log_sigma_weights, encoder_out_log_sigma_biases):

        print("Input shape to encoder network: "),
        print(X.get_shape())

        # Encoder network
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.

        layers = []
        print("Adding layer 0")
        layer_0 = self.transfer_fct(tf.add(tf.matmul(X, encoder_weights[0]),
                                           encoder_biases[0]))
        layers.append(layer_0)

        for i in xrange(0, len(encoder_weights) - 1):
            print("Adding layer " + str(i + 1) + "")
            layer_i = self.transfer_fct(tf.add(tf.matmul(layers[i], encoder_weights[i + 1]),
                                           encoder_biases[i + 1]))
            layers.append(layer_i)

        z_mean = tf.add(tf.matmul(layers[len(encoder_weights) - 1], encoder_out_mean_weights),
                        encoder_out_mean_biases)
        z_log_sigma_sq = tf.add(tf.matmul(layers[len(encoder_weights) - 1], encoder_out_log_sigma_weights),
                                encoder_out_log_sigma_biases)

        # Draw one sample z from Gaussian distribution
        with tf.name_scope("sample_gaussian"):
            n_z = self.latent_dimensions
            eps = tf.random_normal((self.batch_size, n_z), 0, 1, dtype=tf.float32)

        # z = mu + sigma*epsilon
        z = tf.add(z_mean,
                   tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))

        return z_mean, z_log_sigma_sq, z



    def partial_fit(self, X):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        # opt, cost, latent_space_representation = self.sess.run((self.optimizer, self.cost, self.z),
        #                           feed_dict={self.x: X})

        # opt, cost, x, x_r = self.sess.run((self.optimizer, self.cost, self.x, self.x_reconstructed_mean),
        #                           feed_dict={self.x: X})

        # opt, cost = self.sess.run((self.optimizer, self.cost),
        #                            feed_dict={self.x: X})

        opt, cost, rec_loss, lat_loss, def_loss = self.sess.run((self.optimizer, self.cost, self.r_l, self.l_l, self.d_l),
                                  feed_dict={self.x: X})

        # print("x=" + str(x) + ",  x_r=" + str(x_r))
        # print("LSR: "),
        # print(str(latent_space_representation))

        # print("Reconstruction loss: " + str(rec_loss) + "; latent loss: " + str(lat_loss) + "; deformation loss: " + str(def_loss) + ".")

        # return cost
        return cost, rec_loss, lat_loss, def_loss

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
            z_mu = np.random.normal(size=self.latent_dimensions)
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstructed_mean,
                             feed_dict={self.z: z_mu})

    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstructed_mean,
                             feed_dict={self.x: X})