import numpy as np

import tensorflow as tf


class VariationalAutoencoder(object):
    """Variational Autoencoder (VAE).

    Some of this code is based on examples from the following two websites:
        * https://jmetzen.github.io/2015-11-27/vae.html
        * https://github.com/fastforwardlabs/vae-tf

    This implementation uses probabilistic encoders and decoders using Gaussian
    distributions and realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.

    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """

    def __init__(self, network_architecture, sess, transfer_fct=tf.nn.softplus,
                 learning_rate=0.001, batch_size=100):
        # Get all parameters from input
        self.network_architecture = network_architecture
        self.sess = sess
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Placeholder for tensorflow graph input  TODO: Instead of a dict, make this a class
        self.x = tf.placeholder(tf.float32, [None, network_architecture["num_input_neurons"]])

        # Create autoencoder network
        self._create_network()

        # Create loss optimizer based on reconstruction loss and KL divergence
        self._create_loss_optimizer()


        # Add a logger
        self.logger = tf.summary.FileWriter('./log', self.sess.graph)


        # Initialize all variables!
        self.sess.run(tf.global_variables_initializer())


    def _create_network(self):
        # Initialize all weights and biases in the autoencoder network
        network_weights = self._initialize_weights(**self.network_architecture)

        # Use the recognition network to determine the mean and
        # the )log) variance of the Gaussian distribution in
        # latent space
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(network_weights["recognition_network_weights"],
                                      network_weights["recognition_network_biases"])

        # Draw one sample z from Gaussian distribution
        with tf.name_scope("sample_gaussian"):
            n_z = self.network_architecture["n_z"]
            eps = tf.random_normal((self.batch_size, n_z), 0, 1, dtype=tf.float32)

        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean,
                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstructed_mean = \
            self._generator_network(network_weights["generator_network_weights"],
                                    network_weights["generator_network_biases"])


    def _recognition_network(self, weights, biases):
        with tf.name_scope("recognition_network"):

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

        return z_mean, z_log_sigma_sq

    def _generator_network(self, weights, biases):
        with tf.name_scope("generator_network"):

            # Generate probabilistic decoder (decoder network), which
            # maps points in latent space onto a Bernoulli distribution in data space.
            # The transformation is parametrized and can be learned.
            layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']),
                                               biases['b1']))
            layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                               biases['b2']))
            x_reconstructed_mean = \
                tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']),
                                     biases['out_mean']))

        return x_reconstructed_mean

    def _create_loss_optimizer(self):
        offset = tf.constant(1e-5)

        with tf.name_scope("cost"):
            # The loss is composed of two terms:
            # 1.) The reconstruction loss (the negative log probability
            #     of the input under the reconstructed Bernoulli distribution
            #     induced by the decoder in the data space).
            #     This can be interpreted as the number of "nats" required
            #     for reconstructing the input when the activation in latent
            #     is given.
            # Adding 1e-10 to avoid evaluation of log(0.0)
            # reconstruction_loss = \
            #     -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstructed_mean)
            #                    + (1 - self.x) * tf.log(1e-10 + 1 - self.x_reconstructed_mean),
            #                    1)

            reconstruction_loss = \
                -tf.reduce_sum(tf.add(tf.multiply(self.x, tf.log(tf.add(offset, self.x_reconstructed_mean))),
                               tf.multiply((tf.subtract(tf.constant(1.0), self.x)),
                               tf.log(tf.add(tf.constant(1.0), tf.subtract(offset, self.x_reconstructed_mean))))),
                               axis=1)

            self.r_loss = reconstruction_loss
            # 2.) The latent loss, which is defined as the Kullback-Leibler divergence
            ##    between the distribution in latent space induced by the encoder on
            #     the data and some prior. This acts as a kind of regularizer.
            #     This can be interpreted as the number of "nats" required
            #     for transmitting the the latent space distribution given
            #     the prior.
            # latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
            #                                    - tf.square(self.z_mean)
            #                                    - tf.exp(self.z_log_sigma_sq), 1)

            latent_loss = tf.subtract(tf.constant(0.0), tf.multiply(tf.constant(0.5),
                                tf.reduce_sum(tf.subtract(tf.subtract(tf.add(tf.constant(1.0), self.z_log_sigma_sq),
                                               tf.square(self.z_mean)),
                                               tf.exp(self.z_log_sigma_sq)), 1)))

            self.l_loss = latent_loss

            self.cost = tf.reduce_mean(reconstruction_loss + latent_loss)  # average over batch


        # with tf.name_scope("Adam_optimizer"):
            # Use ADAM optimizer
            # self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            # gradients, variables = zip(*self.optimizer.compute_gradients(self.cost))
            # gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            # self.train_op = self.optimizer.apply_gradients(zip(gradients, variables))

        with tf.name_scope("Adam_optimizer"):
            # Use ADAM optimizer
            self.optimizer = \
                tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
            self.train_op = self.optimizer

    def partial_fit(self, X):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        # opt, cost = self.sess.run((self.optimizer, self.cost),
        #                           feed_dict={self.x: X})

        opt, cost = self.sess.run((self.train_op, self.cost),
                                  feed_dict={self.x: X})


        if np.isnan(cost):
            print("Ok. The current overall cost is nan.")
            print("So, let's get the current average reconstruction and latent costs for this batch.")
            print("Average reconstruction loss: " +
                  str(self.sess.run(tf.reduce_mean(self.r_loss),
                                    feed_dict={self.x: X})) +
                  "; Average latent loss: " +
                  str(self.sess.run(tf.reduce_mean(self.l_loss),
                                    feed_dict={self.x: X})) +
                  ".")

        # if r_l:
        #     print("Average reconstruction loss: " + str(self.sess.run(tf.reduce_mean(self.r_loss), feed_dict={self.x: X})) + "; "
        #           "Average latent loss: " + str(self.sess.run(tf.reduce_mean(self.l_loss), feed_dict={self.x: X})) + ".")

        return cost, self.r_loss, self.l_loss

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
        return self.sess.run(self.x_reconstructed_mean,
                             feed_dict={self.z: z_mu})

    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstructed_mean,
                             feed_dict={self.x: X})


    def save(self):
        self.logger.flush()
        self.logger.close()


    @staticmethod
    def _initialize_weights(num_neurons_recognition_layer_1, num_neurons_recognition_layer_2,
                            num_neurons_generator_layer_1, num_neurons_generator_layer_2,
                            num_input_neurons, n_z):
        """
        Initializes all of the weights in the network, and returns a combined dictionary.

        :param num_neurons_recognition_layer_1: Number of 1st layer encoder neurons
        :param num_neurons_recognition_layer_2: Number of 2nd layer encoder neurons
        :param num_neurons_generator_layer_1: Number of 1st layer decoder neurons
        :param num_neurons_generator_layer_2: Number of 2nd layer decoder neurons
        :param num_input_neurons: The data input shape (e.g. 784 for MNIST)
        :param n_z: The number of latent space dimensions
        :return: A dictionary of initialized weights
        """
        all_weights = dict()

        all_weights['recognition_network_weights'] = {
            'h1'           : tf.Variable(xavier_init(num_input_neurons, num_neurons_recognition_layer_1)),
            'h2'           : tf.Variable(xavier_init(num_neurons_recognition_layer_1, num_neurons_recognition_layer_2)),
            'out_mean'     : tf.Variable(xavier_init(num_neurons_recognition_layer_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(num_neurons_recognition_layer_2, n_z))}

        all_weights['recognition_network_biases'] = {
            'b1'           : tf.Variable(tf.zeros([num_neurons_recognition_layer_1], dtype=tf.float32)),
            'b2'           : tf.Variable(tf.zeros([num_neurons_recognition_layer_2], dtype=tf.float32)),
            'out_mean'     : tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}

        all_weights['generator_network_weights'] = {
            'h1'           : tf.Variable(xavier_init(n_z, num_neurons_generator_layer_1)),
            'h2'           : tf.Variable(xavier_init(num_neurons_generator_layer_1, num_neurons_generator_layer_2)),
            'out_mean'     : tf.Variable(xavier_init(num_neurons_generator_layer_2, num_input_neurons)),
            'out_log_sigma': tf.Variable(xavier_init(num_neurons_generator_layer_2, num_input_neurons))}

        all_weights['generator_network_biases'] = {
            'b1'           : tf.Variable(tf.zeros([num_neurons_generator_layer_1], dtype=tf.float32)),
            'b2'           : tf.Variable(tf.zeros([num_neurons_generator_layer_2], dtype=tf.float32)),
            'out_mean'     : tf.Variable(tf.zeros([num_input_neurons], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([num_input_neurons], dtype=tf.float32))}

        return all_weights



def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)



