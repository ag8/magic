import numpy as np

import tensorflow as tf


# def random_rotation_matrix():
#     a = tf.random_uniform(shape=[], minval=0, maxval=2 * np.pi)
#     b = [[tf.cos(a), tf.sin(a)], [tf.subtract(tf.constant(0, dtype=tf.float32), tf.sin(a)), tf.cos(a)]]
#     c = tf.convert_to_tensor(b)
#
#     return c
#
#
# r1 = random_rotation_matrix()
# r2 = random_rotation_matrix()
# r3 = random_rotation_matrix()
#
# with tf.Session() as sess:
#     print(sess.run(r1))
#     print(sess.run(r2))
#     print(sess.run(r3))


# a = [[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]], [[7, 7, 7], [8, 8, 8], [9, 9, 9]], [[10, 10, 10], [11, 11, 11], [12, 12, 12]], [[13, 13, 13], [14, 14, 14], [15, 15, 15]]]

a = np.zeros(shape=[100, 200, 200, 2])

print(np.shape(a))
print(np.shape(a[0, :, :, 0]))
