from __future__ import print_function

import os

import hashlib

import numpy as np

from scipy.misc import imread

def generate_lock_hashes(data_dir):
    """
    Warning: use this function only for debugging on small datasets.
    Using this function on large datasets will result in huge slowdowns.
    """

    print("GENERATING HASHES. ############################################")

    hashes = []

    for i in xrange(0, 100):
        lock_image_file = os.path.join(data_dir, '%d_L.png' % i)

        lock_image = imread(lock_image_file)
        lock_image /= 255

        hashes.append(lock_hash(lock_image))

    print("DONE GENERATING HASHES. #######################################")

    return hashes


def lock_hash(im_array):
    unrolled_image = np.reshape(im_array, newshape=(200 * 200))
    lock_string = ''.join(str(int(e)) for e in unrolled_image)

    print(sum(unrolled_image))

    lock_string_hash = hashlib.sha256(lock_string.encode('utf-8')).hexdigest()

    return lock_string_hash
