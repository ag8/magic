import pickle

import numpy as np

a = np.zeros([24, 200, 200, 3])

print(np.shape(a[10, :, :, 0]))
print(np.shape(a[10, :, :, 1]))
print(np.shape(a[10, :, :, 2]))


# for i in xrange(0, 2):
#     print(i)


with open('/data/affinity/2d/overlap_tiny/OVERLAP_AREAS') as fp:
    overlap_areas = pickle.load(fp)
    print(overlap_areas[:50])
