import sys
import os

import tensorflow as tf

from constants import FLAGS
from constants import DISTANCE_METRICS

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils import *


class TangoEncoder(object):
    """
    The TangoEncoder is the magic autoencoder architecture that learns
    the latent space in a particular way to solve the task at hand.

    A fair amount of the logic here was adapted from Jan Hendrik Metzen
    (https://jmetzen.github.io/2015-11-27/vae.html)
    """

    def __init__(self, sess):
        # First, set the session, to keep
        # all calculations in the same place
        self.sess = sess

        # Determine network architecture

        # The encoder can have as many layers as we want
        self.encoder_neurons = [500, 500]
        assert len(self.encoder_neurons) > 0

        # The decoder can only have two layers, because
        # I'm lazy and don't want to implement another
        # layer for loop
        self.decoder_neurons = [500, 500]
        assert len(self.decoder_neurons) == 2

        # Get the number of input neurons.
        # This is the number of pixels in an image,
        # multiplied by the number of images we
        # stack together at once (assuming each
        # image is B/W--that is, each image has
        # exactly one color channel)
        self.num_input_neurons = FLAGS.IMAGE_SIZE * FLAGS.IMAGE_SIZE * FLAGS.NUM_LAYERS

        # Get the number of dimensions in the
        # latent space (n_z)
        self.latent_dimensions = 80

        # Other network hyperparameters

        # The transfer function to use
        # during training (usually tanh
        #   or softplus)
        # WAIT except softmax just magically works?!
        self.transfer_fct = tf.nn.softmax

        # The learning rates for the two
        # AdamOptimizers. The first learning
        # rate is for the latent space
        # generation; the second is for the
        # metricization of the latent space.
        self.learning_rates = [1e-4, 1e-2]

        # The batch size for training
        self.batch_size = FLAGS.BATCH_SIZE

        # Now, we create a placeholder for
        # the input data, as well as the
        # labels (for metric learning)
        self.x = tf.placeholder(tf.float32, [None, self.num_input_neurons])
        self.overlap_areas = tf.placeholder(tf.float32, [None])

        # Now, we build the network architecture

        # First, the number of input neurons in each
        # network is the number of total inputs
        # divided by the number of images in the input,
        # since we have a separate network for each
        # of the lock, rotated lock, and key images
        self.num_input_neurons_in_network = self.num_input_neurons / FLAGS.NUM_LAYERS

        # START ENCODER WEIGHTS-----------------------------------------------------------------------------------------
        # Here, we build the layers
        # of the encoder network.

        num_encoder_layers = len(self.encoder_neurons)

        encoder_weights = []

        # Add zeroth layer
        connection_info(self.num_input_neurons_in_network, self.encoder_neurons[0])
        encoder_h0_weights = tf.Variable(xavier_init(self.num_input_neurons_in_network, self.encoder_neurons[0]))
        encoder_weights.append(encoder_h0_weights)

        # Add as many layers as required
        for i in xrange(0, num_encoder_layers - 1):
            connection_info(self.encoder_neurons[i], self.encoder_neurons[i + 1])
            encoder_hi_weights = tf.Variable(xavier_init(self.encoder_neurons[i], self.encoder_neurons[i + 1]))
            encoder_weights.append(encoder_hi_weights)

        # Add the last layers
        encoder_out_mean_weights = tf.Variable(
            xavier_init(self.encoder_neurons[num_encoder_layers - 1], self.latent_dimensions))
        encoder_out_log_sigma_weights = tf.Variable(
            xavier_init(self.encoder_neurons[num_encoder_layers - 1], self.latent_dimensions))

        # END ENCODER WEIGHTS-------------------------------------------------------------------------------------------


        # START ENCODER BIASES------------------------------------------------------------------------------------------
        # Basically the same thing
        # as above, but with biases

        encoder_biases = []

        # Add zeroth layer
        bias_info(self.encoder_neurons[0])
        encoder_b0_biases = tf.Variable(tf.zeros([self.encoder_neurons[0]], dtype=tf.float32))
        encoder_biases.append(encoder_b0_biases)

        for i in xrange(1, num_encoder_layers):
            bias_info(self.encoder_neurons[i])
            encoder_bi_biases = tf.Variable(tf.zeros([self.encoder_neurons[i]], dtype=tf.float32))
            encoder_biases.append(encoder_bi_biases)

        # Add the last layers
        encoder_out_mean_biases = tf.Variable(tf.zeros([self.latent_dimensions], dtype=tf.float32))
        encoder_out_log_sigma_biases = tf.Variable(tf.zeros([self.latent_dimensions], dtype=tf.float32))

        # END ENCODER BIASES--------------------------------------------------------------------------------------------


        # START ENCODER WEIGHTS AND BIASES------------------------------------------------------------------------------
        # Now, we build the two-layer
        # decoder network. TODO: Implement an arbitrary-layer decoder, similarly to the encoder above

        decoder_h1_weights = tf.Variable(xavier_init(self.latent_dimensions, self.decoder_neurons[0]))
        decoder_h2_weights = tf.Variable(xavier_init(self.decoder_neurons[0], self.decoder_neurons[1]))
        decoder_out_mean_weights = tf.Variable(xavier_init(self.decoder_neurons[1], self.num_input_neurons_in_network))
        decoder_out_log_sigma_weights = tf.Variable(
            xavier_init(self.decoder_neurons[1], self.num_input_neurons_in_network))

        decoder_b1_biases = tf.Variable(tf.zeros([self.decoder_neurons[0]], dtype=tf.float32))
        decoder_b2_biases = tf.Variable(tf.zeros([self.decoder_neurons[1]], dtype=tf.float32))
        decoder_out_mean_biases = tf.Variable(tf.zeros([self.num_input_neurons_in_network], dtype=tf.float32))
        decoder_out_log_sigma_biases = tf.Variable(tf.zeros([self.num_input_neurons_in_network], dtype=tf.float32))

        # END ENCODER WEIGHTS AND BIASES--------------------------------------------------------------------------------


        # Now that we've initialized our network weights,
        # we're going to do some things with the input
        # and launch them into a network we build (in
        # another function)


        # First, let's reshape the input.
        # This is because we (might) get the input
        # as a long tensor with shape
        # [BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE * NUM_LAYERS].
        # We want to be able to separate the three
        # images in the input, so we reshape the tensor
        # into the shape
        # [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_LAYERS].
        self.reshaped_x = tf.reshape(self.x, [-1, FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, FLAGS.NUM_LAYERS])

        # Now, we split by the third axis
        # into three parts to obtain the
        # lock image, rotated lock image,
        # and the key image.
        lock_image, rotated_lock_image, key_image = split(self.reshaped_x, num_splits=3)

        # Now, we reshape each of the three images
        # into long vector batches (so tensors of
        # size [BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE]
        lock_image = tf.reshape(lock_image, [-1, FLAGS.IMAGE_SIZE * FLAGS.IMAGE_SIZE])
        rotated_lock_image = tf.reshape(rotated_lock_image, [-1, FLAGS.IMAGE_SIZE * FLAGS.IMAGE_SIZE])
        key_image = tf.reshape(key_image, [-1, FLAGS.IMAGE_SIZE * FLAGS.IMAGE_SIZE])

        # We will also need the lock image vector later,
        # when calculating the reconstruction loss.
        # We won't need any of the other image vectors for
        # a really specific reason (see the comments
        # in the optimizer code below for details).
        self.lock_image = lock_image


        # Now, we're ready to feed our images
        # into the actual network!


        # First, let's combine the weights and biases
        # of our encoder network into one array, since
        # we have to pass it three times as a parameter
        encoder_weights_and_biases = [encoder_weights, encoder_biases,
                              encoder_out_mean_weights, encoder_out_mean_biases,
                              encoder_out_log_sigma_weights, encoder_out_log_sigma_biases]

        # Now, let's get the z mean, the z log sigma squared,
        # and the z itself (i.e., the latent space representation)
        # of the lock image.
        self.z_mean, self.z_log_sigma_sq, self.z = self.get_latent_representation(lock_image,
                                                                                  encoder_weights_and_biases)

        # Now, get the latent space representation of the rotated lock image.
        # We only need the z_mean and the z_log_sigma_sq for one of the three
        # images for good latent learning, so we ignore them here.
        _, _, self.z_representation_of_rotated = self.get_latent_representation(rotated_lock_image,
                                                                                encoder_weights_and_biases)

        # Similarly, get the latent space representation of the key image.
        _, _, self.z_representation_of_key = self.get_latent_representation(key_image,
                                                                            encoder_weights_and_biases)


        # Now, let's build a decoder network.
        # This probabilistic decoder maps points from latent space
        # onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, decoder_h1_weights),
                                           decoder_b1_biases))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, decoder_h2_weights),
                                           decoder_b2_biases))
        self.x_reconstructed_mean = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, decoder_out_mean_weights),
                                                         decoder_out_mean_biases))

        # Now, let's create the loss optimizer.
        # We have the whole network ready to go,
        # so now we just need to know what our
        # loss is, and what it is we want to
        # minimize.
        with tf.name_scope("cost"):
            # The loss is composed of two terms:
            # 1.) The reconstruction loss (the negative log probability
            #     of the input under the reconstructed Bernoulli distribution
            #     induced by the decoder in the data space).
            #     This can be interpreted as the number of "nats" required
            #     for reconstructing the input when the activation in latent
            #     is given.
            # Adding 1e-10 to avoid evaluation of log(0.0)
            #
            # REMARK: When doing metric learning, the reconstruction
            #         loss imposes a fundamentally contradictory constraint.
            #         Consider this: we're trying to learn a rotation-invariant
            #         representation of images in the latent space; however,
            #         if we have a reconstruction loss, it will force a
            #         distinction between the latent space representations between
            #         the lock image and the rotated lock image, which is the opposite
            #         of what we are minimizing elsewhere. Hence, we calculate
            #         the reconstruction loss _only_ on the non-rotated lock image.
            #         Note that above, our generator network only reconstructs
            #         the non-rotated lock image, because that is all we care about
            #         being able to reconstruct accurately.

            # in_reduce_sum = self.x * tf.log(1e-10 + self.x_reconstructed_mean) + (1 - self.x) * tf.log(1e-10 + 1 - self.x_reconstructed_mean)
            # in_reduce_sum = tf.log(tf.pow(self.x_reconstructed_mean, self.x)) + (1.0 - self.x) * tf.log(1.0 - self.x_reconstructed_mean)
            # in_reduce_sum = tf.log(tf.multiply(tf.pow(self.x_reconstructed_mean, self.x), tf.pow(1.0 - self.x_reconstructed_mean, 1.0 - self.x)))

            in_reduce_sum = tf.log(tf.multiply(tf.pow(self.x_reconstructed_mean, self.lock_image),
                                               tf.pow(1.0 - self.x_reconstructed_mean, 1.0 - self.lock_image)))

            reconstruction_loss = -tf.reduce_sum(in_reduce_sum, 1)

            # Set a field variable to the reconstruction loss
            # so we can access it later (for info purposes)
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
            #     representations of the lock image. Having this loss
            #     tells the network that latent space representations
            #     of an image and the same image, but rotated, need to
            #     be the same--thus, we are forcing transformation
            #     invariance on our latent space.
            deformation_loss = 100 * tf.reduce_sum(tf.squared_difference(self.z, self.z_representation_of_rotated), 1)
            self.d_l = tf.reduce_mean(deformation_loss)

            # We add our three losses together, and average over the batch,
            # to get our final cost function.
            self.cost = tf.reduce_mean(reconstruction_loss + latent_loss + deformation_loss)  # average over batch


            # Now, the actual loss for training--i.e., the metric learning cost.
            # We say that the distance between the latent space representations
            # of our lock and key image should be equal to their maximum overlap
            # area. So, we define the loss as the (squared) difference between
            # the distance between the two vectors, and the actual overlap area.
            if FLAGS.DISTANCE_METRIC == DISTANCE_METRICS.COSINE_DISTANCE:
                self.distance = tf.reciprocal(cosine_distance(self.z, self.z_representation_of_key))
            elif FLAGS.DISTANCE_METRIC == DISTANCE_METRICS.SQUARED_DIFFERENCE:
                self.distance = tf.reciprocal(tf.reduce_sum(tf.squared_difference(self.z, self.z_representation_of_key), 1))
            else:
                raise ValueError("Invalid distance metric: " + str(FLAGS.DISTANCE_METRIC))

            self.training_loss = tf.reduce_mean(tf.squared_difference(self.distance, self.overlap_areas))


        # Now, we just have to create the optimizers
        # to actually minimize over our losses.

        # First minimizer: the latent space learning cost
        with tf.name_scope("Adam_optimizer"):
            self.optimizer = \
                tf.train.AdamOptimizer(learning_rate=self.learning_rates[0]).minimize(self.cost)

        # Second minimizer: metric learning loss
        with tf.name_scope("Adam_optimizer_for_overlap"):
            self.o_optimizer = \
                tf.train.AdamOptimizer(learning_rate=self.learning_rates[1]).minimize(self.training_loss)

        # Ok, now we're done with everything.

        # Initialize all variables!
        self.sess.run(tf.global_variables_initializer())


    def get_latent_representation(self, X, weights_and_biases):

        # Get all weights/biases data from array
        encoder_weights = weights_and_biases[0]
        encoder_biases = weights_and_biases[1]
        encoder_out_mean_weights = weights_and_biases[2]
        encoder_out_mean_biases = weights_and_biases[3]
        encoder_out_log_sigma_weights = weights_and_biases[4]
        encoder_out_log_sigma_biases = weights_and_biases[5]


        # Encoder network
        # Generate a probabilistic encoder that maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.

        layers = []

        # Add the zeroth layer
        layer_0 = self.transfer_fct(tf.add(tf.matmul(X, encoder_weights[0]),
                                           encoder_biases[0]))
        layers.append(layer_0)

        # Add as many layers to the encoder as necessary
        for i in xrange(0, len(encoder_weights) - 1):
            layer_i = self.transfer_fct(tf.add(tf.matmul(layers[i], encoder_weights[i + 1]),
                                               encoder_biases[i + 1]))
            layers.append(layer_i)

        # Add the last layers
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

        # Return the z mean, z log sigma squared,
        # as well as the latent space repsesentation itself
        return z_mean, z_log_sigma_sq, z



    def partial_fit(self, X, overlap_areas):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        # opt, cost, latent_space_representation = self.sess.run((self.optimizer, self.cost, self.z),
        #                           feed_dict={self.x: X})

        # opt, cost, x, x_r = self.sess.run((self.optimizer, self.cost, self.x, self.x_reconstructed_mean),
        #                           feed_dict={self.x: X})

        # opt, cost = self.sess.run((self.optimizer, self.cost),
        #                            feed_dict={self.x: X})

        opt, o_opt, cost, training_loss, rec_loss, lat_loss, def_loss, distance = self.sess.run(
            (self.optimizer, self.o_optimizer, self.cost, self.training_loss, self.r_l, self.l_l, self.d_l, self.distance),
            feed_dict={self.x            : X,
                       self.overlap_areas: overlap_areas})

        # print("x=" + str(x) + ",  x_r=" + str(x_r))
        # print("LSR: "),
        # print(str(latent_space_representation))

        # print("Reconstruction loss: " + str(rec_loss) + "; latent loss: " + str(lat_loss) + "; deformation loss: " + str(def_loss) + ".")

        # return cost
        return cost, training_loss, rec_loss, lat_loss, def_loss, distance

    def get_predictions(self, X, overlap_areas):
        return self.sess.run(self.distance, feed_dict={self.x: X, self.overlap_areas: overlap_areas})

    def transform(self, X, overlap_areas):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X, self.overlap_areas: overlap_areas})

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

    def reconstruct(self, X, overlap_areas):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstructed_mean,
                             feed_dict={self.x            : X,
                                        self.overlap_areas: overlap_areas})



def cosine_distance(a, b):
    """
    Calculates the cosine distance between two vectors,
    after automatic l2 regulatization.

    Based on https://stackoverflow.com/a/43358711/

    :param a: the first vector
    :param b: the second vector
    :return: the cosine distance between a and b
    """
    normalize_a = tf.nn.l2_normalize(a, 0)
    normalize_b = tf.nn.l2_normalize(b, 0)
    cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b), 1)
    return cos_similarity