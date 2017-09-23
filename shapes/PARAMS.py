from constants import FLAGS

# Global constants describing the MSHAPES data set.
IMAGE_SIZE = FLAGS.IMAGE_SIZE
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# How many examples to use for training in the queue
NUM_EXAMPLES_TO_LOAD_INTO_QUEUE = FLAGS.NUM_EXAMPLES_TO_LOAD_INTO_QUEUE

DATA_DIR = FLAGS.DATA_DIR

# Where to download the MSHAPES dataset from
DATA_URL = 'https://electronneutrino.com/affinity/shapes/datasets/MSHAPES_180DEG_ONECOLOR_SIMPLE_50k_100x100.zip'
