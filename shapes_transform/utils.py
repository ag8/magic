#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pwd
import socket
import sys
import urllib
import zipfile
from glob import glob
from time import strftime, gmtime

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

import debugutils as dbu

import requests
# from PIL import Image
from six.moves import urllib as smurllib

import tensorflow as tf

from constants import FLAGS


def maybe_download_and_extract(data_url, data_dir, allow_downloads=True):
    """
    Downloads and extracts the data zip, if necessary

    :param data_url:  the url from which to download the data
    :param data_dir: the folder to store the data in. If there is data here already, the download is cancelled.
    :param allow_downloads: whether to allow downloads
    :return:
    """

    # Nothing to do here if downloads aren't allowed
    if not allow_downloads:
        return

    dest_directory = data_dir

    # Create the destination directory
    # if they do not already exist
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)

    # If the files don't already exist,
    # download the data as necessary
    if not os.path.exists(filepath):
        print_progress_bar(0, 100,
                           prefix='Downloading ' + filename + ":", suffix='Complete', length=50,
                           fill='█')

        def _progress(count, block_size, total_size):
            print_progress_bar(float(count * block_size) / float(total_size) * 100.0, 100,
                               prefix='Downloading ' + filename + ":", suffix='Complete', length=50,
                               fill='█')

        filepath, _ = smurllib.request.urlretrieve(data_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    extracted_dir_path = os.path.join(dest_directory, '/images')

    # Extract the zip, unless the extracted path
    # already exists
    if not os.path.exists(extracted_dir_path):
        zip_ref = zipfile.ZipFile(filepath, 'r')
        zip_ref.extractall(dest_directory)
        zip_ref.close()


def get_time_string():
    """
    Returns the GMT.

    :return: a formatted string containing the GMT.
    """
    return strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " GMT"


def get_username():
    """
    Gets the username of the current user.

    :return: a string with the username
    """
    return pwd.getpwuid(os.getuid()).pw_name


def get_hostname():
    """
    Returns the hostname of the computer.

    :return: a string containing the hostname
    """
    return socket.gethostname()


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar. Based on https://stackoverflow.com/a/34325723
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix) + '\r')
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total:
        print("")


def die(message="", error_code=1):
    """
    Immediately exit the program.

    :param message: (optional) a message to write to stderr
    :param error_code: the error code with which to exit
    :return:
    """
    sys.stderr.write(message)
    sys.exit(error_code)


class BColors:
    """
    A class containing ANSI escape sequencing for output formatting.
    """

    def __init__(self):
        pass

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[43m\033[30m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    MAJORINFO = '\033[93m'


def info(message, source=""):
    """
    Print an info message, if necessary.

    :param message: the message to print
    :param source: (Optional) prefix to prepend to message
    :return:
    """
    if source != "":
        print(BColors.OKBLUE + "[" + source + "]: " + message + "" + BColors.ENDC)
    else:
        print(BColors.OKBLUE + "" + message + "" + BColors.ENDC)


def printi(message):
    """
    Print something in a nice yellow color.

    :param message: the message to print
    :return:
    """
    print(BColors.MAJORINFO + "" + message + "" + BColors.ENDC)


def header(message):
    """
    Print a message in a nice purple color.

    :param message: the message to print
    :return:
    """
    print(BColors.HEADER + "" + BColors.BOLD + "" + message + "" + BColors.ENDC)


def ok(message):
    """
    Print a message in a nice green color.

    :param message: the message to print
    :return:
    """
    print(BColors.OKGREEN + "" + message + "" + BColors.ENDC)


def warning(message):
    """
    Print a message in a nice warning color.

    :param message: the message to print
    :return:
    """
    print(BColors.WARNING + "" + message + "" + BColors.ENDC)


def majorwarning(message):
    """
    Print a message in a nice failure color.

    :param message: the message to print
    :return:
    """
    print(BColors.FAIL + "" + "" + message + "" + BColors.ENDC)


def error(message):
    """
    Print a message in a nice failure color.

    @alias majorwarning(message)

    :param message: the message to print
    :return:
    """
    print(BColors.FAIL + "" + message + "" + BColors.ENDC)


def check_dataset(data_dir):
    """
    Makes sure the dataset seems ok.

    :param data_dir: the directory with the av4 data
    :raises Exception if dataset contains no av4 files
    :return:
    """
    if len(glob(os.path.join(data_dir + '/**/', "*[_]*.av4"))) == 0:
        raise Exception("Dataset appears to be empty (looking in " + data_dir + ")")


def rightzpad(string, length):
    """
    Pad a string from the right to a given length with zeroes.

    :param string: the string to pad
    :param length: the length to pad to
    :return: the padded string
    """
    return (str(string)[::-1]).zfill(length)[::-1]


def send_data_to_poor_mans_tensorboard(value):
    """
    Send a value to Poor Man's Tensorboard on electronneutrino.

    :param value: the value to send
    :return: the response of the server
    """
    params = {'add': value}
    encoded_params = urllib.urlencode(params)

    response = 'No response :('
    response = requests.get(
        'https://electronneutrino.com/affinity/poor%20man%27s%20tensorboard/add.php?' + encoded_params)
    # print (response.status_code)
    # print (response.content)

    return response


def notify(message, subject="Notification", email='andrew2000g@gmail.com'):
    """
    Send an email with the specified message.

    :param message: the message to be sent
    :param subject: (optional) the subject of the message
    :param email: (optional) the email to send the message to. Defaults to FLAGS.NOTIFICATION_EMAIL

    :return: The response of the server. Should be "Thanks!"
    """
    params = {'message': "[" + get_username() + "@" + get_hostname() + ", " + get_time_string() + "]: " + message,
              'subject': subject, 'email': email}
    encoded_params = urllib.urlencode(params)

    response = requests.get('https://electronneutrino.com/affinity/notify/notify.php?' + encoded_params)
    # print (response.status_code)
    # print (response.content)

    return response


def file_exists(filepath):
    """
    Check whether a file exists at the specified filepath.

    :param filepath: the filepath to check
    :return: boolean, whether the file at the filepath exists
    """
    return os.path.isfile(filepath)


def float_to_str(f):
    """
    Converts a float to a string.

    :param f: the float value
    :return: a string with the float value
    """

    float_string = repr(f)
    if 'e' in float_string:  # detect scientific notation
        digits, exp = float_string.split('e')
        digits = digits.replace('.', '').replace('-', '')
        exp = int(exp)
        zero_padding = '0' * (abs(int(exp)) - 1)  # minus 1 for decimal point in the sci notation
        sign = '-' if f < 0 else ''
        if exp > 0:
            float_string = '{}{}{}.0'.format(sign, digits, zero_padding)
        else:
            float_string = '{}0.{}{}'.format(sign, zero_padding, digits)
    return float_string



def connection_info(num_neurons_from, num_neurons_to):
    """
    Print information about connecting network layers.

    :param num_neurons_from: the number of neurons in the (n-1)st layer
    :param num_neurons_to: the number of neurons in the nth layer
    """
    print("Connecting network layer [" + str(num_neurons_from) + "->" + str(num_neurons_to) + "].")


def bias_info(num_neurons):
    """
    Print information about adding biases.

    :param num_neurons: the number of neurons (equals the number of biases to add)
    """
    print("Adding " + str(num_neurons) + " biases.")



def save_example(accuracy_testing_images, accuracy_testing_overlap_areas, predictions, epoch=None, i=None):
    """
    Save an example of a prediction by the autoencoder.

    :param accuracy_testing_images: the image examples for accuracy testing
    :param accuracy_testing_overlap_areas:  the (overlap) labels for accuracy testing
    :param predictions: the neural network's prediction of the labels
    :param epoch: the epoch of training
    :param i: the step of training within the epoch
    """

    # Reshape images
    reshaped_images = np.reshape(accuracy_testing_images,
                                 newshape=[FLAGS.BATCH_SIZE, FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, FLAGS.NUM_LAYERS])

    ex_lock = reshaped_images[0, :, :, 0]
    scipy.misc.imsave("ex_lock.png", ex_lock)
    # print(np.shape(ex_lock))
    # print(sum(sum(ex_lock)))
    ex_lock_r = reshaped_images[0, :, :, 1]
    ex_key = reshaped_images[0, :, :, 2]

    plt.figure(figsize=(8, 12))
    # print("x_sample[i] shape: "),
    # print(np.shape(x_sample[i]))
    # print("")
    # print("x_reconstruct[i] shape: ")
    # print(np.shape(x_reconstruct[i]))
    # print("")

    plt.subplot(5, 2, 3)
    plt.imshow(ex_lock, vmin=0, vmax=1,
               cmap="gray")
    # plt.imshow(x_sample[i].reshape(200, 200, 1)[:, :, 1], vmin=0, vmax=1, cmap="gray")
    plt.title("Lock image")
    plt.colorbar()
    plt.subplot(5, 2, 4)
    # plt.imshow(x_reconstruct[i].reshape(200, 200, FLAGS.NUM_LAYERS)[:, :, 1], vmin=0, vmax=1, cmap="gray")
    plt.imshow(ex_key.reshape(FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, 1)[:, :, 0], vmin=0, vmax=1, cmap="gray")
    plt.title("Key image")
    plt.colorbar()

    plt.subplot(5, 2, 5)
    plt.text(0.1, 0.9, "Actual overlap: " + str(accuracy_testing_overlap_areas[0]) + ", predicted overlap: " + str(
        predictions[0]), fontsize=12)

    plt.tight_layout()

    plt.savefig('ex' + str(epoch) + '_' + str(i) + '.png')


def reconstruct_input(x_sample, overlap_areas, vae, filename='foo.png'):
    """
    Sample the latent space to create a 2d latent sample.

    :param x_sample:
    :param overlap_areas:
    :param vae:
    :param filename:
    :return:
    """
    # Display the input reconstruction
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
        plt.imshow(x_sample[i].reshape(FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, FLAGS.NUM_LAYERS)[:, :, 0], vmin=0, vmax=1, cmap="gray")
        # plt.imshow(x_sample[i].reshape(200, 200, 1)[:, :, 1], vmin=0, vmax=1, cmap="gray")
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2 * i + 2)
        # plt.imshow(x_reconstruct[i].reshape(200, 200, FLAGS.NUM_LAYERS)[:, :, 1], vmin=0, vmax=1, cmap="gray")
        plt.imshow(x_reconstruct[i].reshape(FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, 1)[:, :, 0], vmin=0, vmax=1, cmap="gray")
        plt.title("Reconstruction")
        plt.colorbar()
    plt.tight_layout()

    plt.savefig(filename)


def sample_latent_space(vae, filename='latent_space_2d_sampling.png'):
    if vae.latent_dimensions == 2:
        nx = ny = 20
        x_values = np.linspace(-3, 3, nx)
        y_values = np.linspace(-3, 3, ny)

        canvas = np.empty((FLAGS.IMAGE_SIZE * ny, FLAGS.IMAGE_SIZE * nx))
        for i, yi in enumerate(x_values):
            for j, xi in enumerate(y_values):
                z_mu = np.array([[xi, yi]] * vae.batch_size)
                x_mean = vae.generate(z_mu)
                canvas[(nx - i - 1) * FLAGS.IMAGE_SIZE:(nx - i) * FLAGS.IMAGE_SIZE, j * FLAGS.IMAGE_SIZE:(j + 1) * FLAGS.IMAGE_SIZE] = x_mean[0].reshape(FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, 1)[:, :,
                                                                                   0]

        plt.figure(figsize=(8, 10))
        Xi, Yi = np.meshgrid(x_values, y_values)
        plt.imshow(canvas, origin="upper", cmap="gray")
        plt.tight_layout()

        plt.savefig(filename)
    else:
        print("Number of latent dimensions≠2")


def print_losses(epoch, training_epochs, total_batch, i, training_loss, cost, rec_loss, lat_loss, def_loss, fill=' '):
    training_loss = round(training_loss, -4)

    print(("Epoch: (" + str(epoch) + "/" + str(training_epochs) + ");").ljust(16, fill) + (" i: (" + str(i) + "/" + str(
        total_batch) + ").").ljust(16, fill)),
    print(("TRAINING LOSS: " + str(float(training_loss))).ljust(35, fill)),
    print(("Current cost: " + str(cost) + "").ljust(30, fill)),
    print(("[Reconstruction loss: " + str(rec_loss) + ";").ljust(32, fill) + (" latent loss: " + str(
        lat_loss) + ";").ljust(25, fill) + ("deformation loss: " + str(def_loss) + ".").ljust(27, fill) + "]")



# OVERLAP UTILS

def split(combined_images, num_splits=2):
    return tf.split(combined_images, num_or_size_splits=num_splits, axis=3)


# TENSORFLOW UTILS

def random_rotation_matrix():
    a = tf.random_uniform(shape=[], minval=0, maxval=2 * np.pi)
    b = [[tf.cos(a), tf.sin(a)], [tf.subtract(tf.constant(0, dtype=tf.float32), tf.sin(a)), tf.cos(a)]]
    c = tf.convert_to_tensor(b)

    return c


def random_rotation(lock_image, image_size):
    print("Lock images batch shape: "),
    print(lock_image.get_shape())

    # Reshape the lock images batch into the correct shape,
    # ([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
    lock_image = tf.reshape(lock_image, shape=[1, image_size, image_size, 1])

    angle = tf.random_uniform(shape=[], minval=0, maxval=2 * np.pi)
    rotated_images = tf.contrib.image.rotate(images=lock_image, angles=angle)

    return rotated_images, angle



def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)



# NN utils

def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)
