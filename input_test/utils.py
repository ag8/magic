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

import requests
# from PIL import Image
from six.moves import urllib as smurllib

import tensorflow as tf


def maybe_download_and_extract(data_url, data_dir, allow_downloads=True):
    """Downloads and extracts the zip from electronneutrino, if necessary"""

    # Nothing to do here if downloads aren't allowed
    if not allow_downloads:
        return

    dest_directory = data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
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

    if not os.path.exists(extracted_dir_path):
        zip_ref = zipfile.ZipFile(filepath, 'r')
        zip_ref.extractall(dest_directory)
        zip_ref.close()


# def verify_dataset():
#     """
#     Verifies the authenticity of the dataset.
#
#     :raises: Exception if the dataset's images are the wrong size.
#
#     :return: nothing on success
#     """
#     which = randint(1, 10000)
#
#     where = os.path.join(FLAGS.DATA_DIR, 'images/%d_L.png' % which)
#
#     im = Image.open(where)
#     width, height = im.size
#
#    # print("w, h: " + str(width) + ", " + str(height))

#    if not (width == FLAGS.IMAGE_SIZE and height == FLAGS.IMAGE_SIZE):
#        raise Exception("Dataset appears to have been corrupted. (Check " + where + ")")


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


def send_cross_entropy_to_poor_mans_tensorboard(cross_entropy):
    params = {'add': cross_entropy}
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
    return os.path.isfile(filepath)


def float_to_str(f):
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
    print("Connecting network layer [" + str(num_neurons_from) + "->" + str(num_neurons_to) + "].")


def bias_info(num_neurons):
    print("Adding " + str(num_neurons) + " biases.")



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



# NN utils

def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)
