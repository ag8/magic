import tensorflow as tf

from affinity.models.magic_autoencoder.utils.utils import *


class TangoEncoder(object):
    def __init__(self):
        # Determine network architecture
        self.encoder_neurons_layer1 = 500
        self.encoder_neurons_layer2 = 500
        self.decoder_neurons_layer1 = 500
        self.decoder_neurons_layer2 = 500

        self.num_input_neurons = 784

        self.latent_dimensions = 2

        self.transfer_fct = tf.nn.tanh
        self.batch_size = 24

        self.x = tf.placeholder(tf.float32, [None, self.num_input_neurons])



        # Define the network architecture

        # Initialize encoder weights  TODO: Convert to slim repeat loop
        encoder_h1_weights = tf.Variable(xavier_init(self.num_input_neurons, self.encoder_neurons_layer1))
        encoder_h2_weights = tf.Variable(xavier_init(self.encoder_neurons_layer1, self.encoder_neurons_layer2))
        encoder_out_mean_weights = tf.Variable(xavier_init(self.encoder_neurons_layer2, self.latent_dimensions))
        encoder_out_log_sigma_weights = tf.Variable(xavier_init(self.encoder_neurons_layer2, self.latent_dimensions))

        encoder_b1_biases = tf.Variable(tf.zeros([self.encoder_neurons_layer1], dtype=tf.float32))
        encoder_b2_biases = tf.Variable(tf.zeros([self.encoder_neurons_layer2], dtype=tf.float32))
        encoder_out_mean_biases = tf.Variable(tf.zeros([self.latent_dimensions], dtype=tf.float32))
        encoder_out_log_sigma_biases = tf.Variable(tf.zeros([self.latent_dimensions], dtype=tf.float32))

        decoder_h1_weights = tf.Variable(xavier_init(self.latent_dimensions, self.encoder_neurons_layer1))
        decoder_h2_weights = tf.Variable(xavier_init(self.encoder_neurons_layer1, self.encoder_neurons_layer2))
        decoder_out_mean_weights = tf.Variable(xavier_init(self.encoder_neurons_layer2, self.num_input_neurons))
        decoder_out_log_sigma_weights = tf.Variable(xavier_init(self.encoder_neurons_layer2, self.num_input_neurons))

        decoder_b1_biases = tf.Variable(tf.zeros([self.encoder_neurons_layer1], dtype=tf.float32))
        decoder_b2_biases = tf.Variable(tf.zeros([self.encoder_neurons_layer2], dtype=tf.float32))
        decoder_out_mean_biases = tf.Variable(tf.zeros([self.num_input_neurons], dtype=tf.float32))
        decoder_out_log_sigma_biases = tf.Variable(tf.zeros([self.num_input_neurons], dtype=tf.float32))


        # Encoder network
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.

        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, encoder_h1_weights),
                                           encoder_b1_biases))

        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, encoder_h2_weights),
                                           encoder_b2_biases))

        self.z_mean = tf.add(tf.matmul(layer_2, encoder_out_mean_weights),
                             encoder_out_mean_biases)
        self.z_log_sigma_sq = tf.add(tf.matmul(layer_2, encoder_out_log_sigma_weights),
                                     encoder_out_log_sigma_biases)



        # Draw one sample z from Gaussian distribution
        with tf.name_scope("sample_gaussian"):
            n_z = self.latent_dimensions
            eps = tf.random_normal((self.batch_size, n_z), 0, 1, dtype=tf.float32)

        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean,
                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))



        # Decoder network
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, decoder_h1_weights),
                                           decoder_b1_biases))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, decoder_h2_weights),
                                           decoder_b2_biases))
        x_reconstructed_mean = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, decoder_out_mean_weights),
                                                    decoder_out_mean_biases))
