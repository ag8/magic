import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from vae import VariationalAutoencoder

# Load MNIST data in a format suited for tensorflow.
# The script input_data is available under this URL:
# https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/input_data.py
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples




def train(network_architecture, sess, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=5):
    vae = VariationalAutoencoder(network_architecture, sess=sess,
                                 # transfer_fct=tf.nn.tanh,
                                 learning_rate=learning_rate,
                                 batch_size=batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size)

            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(avg_cost))
    return vae




with tf.Session() as sess:
    # Define the network architecture
    network_architecture = \
        dict(num_neurons_recognition_layer_1=500,  # 1st layer encoder neurons
             num_neurons_recognition_layer_2=500,  # 2nd layer encoder neurons
             num_neurons_generator_layer_1=500,  # 1st layer decoder neurons
             num_neurons_generator_layer_2=500,  # 2nd layer decoder neurons
             num_input_neurons=784,  # MNIST data input (img shape: 28*28)
             n_z=20)  # dimensionality of latent space

    # Train the autoencoder for 75 epochs
    vae = train(network_architecture, sess=sess, training_epochs=20)

    # Initialize all variables
    # sess.run(tf.global_variables_initializer())


    # Display the input reconstruction
    x_sample = mnist.test.next_batch(100)[0]
    x_reconstruct = vae.reconstruct(x_sample)

    plt.figure(figsize=(8, 12))
    for i in range(5):
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Reconstruction")
        plt.colorbar()
    plt.tight_layout()

    plt.savefig('foo.png')
