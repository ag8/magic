import numpy as np

from utils import *

max_epochs = 100
max_i = 100

for epoch in xrange(0, max_epochs):
    for i in xrange(0, max_i):
        magic_print("Hi!", epoch=epoch, max_epochs=max_epochs, i=i, max_i=max_i)
