#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
import os
import pwd
import socket
import sys
import urllib
import zipfile
from glob import glob
from random import randint
from time import strftime, gmtime

import requests
# from PIL import Image
from six.moves import urllib as smurllib

from constants import FLAGS


def check_dependencies_installed():
    """
    Checks whether the needed dependencies are installed.

    :return: a list of missing dependencies
    """
    missing_dependencies = []

    try:
        import importlib
    except ImportError:
        missing_dependencies.append("importlib")

    dependencies = ["termcolor",
                    "colorama",
                    "tensorflow",
                    "numpy",
                    "PIL",
                    "six",
                    "tarfile",
                    "zipfile",
                    "requests"]

    for dependency in dependencies:
        if not can_import(dependency):
            missing_dependencies.append(dependency)

    return missing_dependencies


def can_import(some_module):
    """
    Checks whether a module is installed by trying to import it.

    :param some_module: the name of the module to check

    :return: a boolean representing whether the import is successful.
    """

    try:
        importlib.import_module(some_module)
    except ImportError:
        return False

    return True


def maybe_download_and_extract():
    """Downloads and extracts the zip from electronneutrino, if necessary"""

    # Nothing to do here if downloads aren't allowed
    if not FLAGS.ALLOW_DOWNLOADS:
        return

    dest_directory = FLAGS.DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = FLAGS.DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        print_progress_bar(0, 100,
                           prefix='Downloading ' + filename + ":", suffix='Complete', length=50,
                           fill='█')

        def _progress(count, block_size, total_size):
            print_progress_bar(float(count * block_size) / float(total_size) * 100.0, 100,
                               prefix='Downloading ' + filename + ":", suffix='Complete', length=50,
                               fill='█')

        filepath, _ = smurllib.request.urlretrieve(FLAGS.DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

        # If it's a fresh dataset, the minimum file number is 1 (per standard enumeration)
        FLAGS.MIN_FILE_NUM = 1

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


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill="█"):
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
        print()
        print()
        sys.stdout.write("")


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
    if FLAGS.PRINT_INFO:
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
    if FLAGS.USE_TENSORBOARD:
        response = requests.get(
            'https://electronneutrino.com/affinity/poor%20man%27s%20tensorboard/add.php?' + encoded_params)
        print (response.status_code)
        print (response.content)

    return response


def notify(message, subject="Notification", email=FLAGS.NOTIFICATION_EMAIL):
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

    response = "Emailing blocked by configuration (See constants.py)"

    if FLAGS.EMAIL_INFO:
        response = requests.get('https://electronneutrino.com/affinity/notify/notify.php?' + encoded_params)
        # print (response.status_code)
        # print (response.content)

    return response


def get_width():
    rows, columns = os.popen('stty size', 'r').read().split()
    return int(columns)


def parade_output(message):
    width = get_width()
    print("")
    sys.stdout.write("%" * width)
    sys.stdout.write("%" + " " * (width - 2) + "%")
    sys.stdout.write("%" + " " * ((width - 2 - message.__len__()) / 2) + message + " " * ((width - 2 - message.__len__()) / 2) + "%")
    sys.stdout.write("%" + " " * (width - 2) + "%")
    sys.stdout.write("%" * width)
    print("")
