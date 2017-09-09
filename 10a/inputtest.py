import numpy as np
import scipy.misc

import pickle

import tensorflow as tf

import overlap_input
from constants import FLAGS

images_batch, labels_batch = overlap_input.inputs(normalize=True, reshape=True, rotation=FLAGS.ROTATE)

with open('/data/affinity/2d/overlap_tiny/OVERLAP_AREAS') as fp:
    overlap_areas = pickle.load(fp)
    print(overlap_areas[:FLAGS.BATCH_SIZE])

with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    tf.global_variables_initializer().run()

    images = images_batch.eval()
    labels = labels_batch.eval()

    print("Labels: " + str(labels))

    # TESTING THAT THE BATCHES ARE CORRECT
    reshaped_images = np.reshape(images,
                                 newshape=[FLAGS.BATCH_SIZE, FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE,
                                           FLAGS.NUM_LAYERS])
    lock_image = reshaped_images[0, :, :, 0]
    lock_image_r = reshaped_images[0, :, :, 1]
    key_image = reshaped_images[0, :, :, 2]

    scipy.misc.imsave('lock.png', lock_image)
    scipy.misc.imsave('key.png', key_image)

    print("Overlap: " + str(labels[0]))

