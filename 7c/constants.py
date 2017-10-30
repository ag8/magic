#!/usr/bin/env python
# -*- coding: utf-8 -*-

class FLAGS:
    def __init__(self):
        pass

    "Dataset info______________________________________________________________________________________________________"

    # Where the dataset should be stored on the computaer.
    DATA_DIR = '/data/affinity/2d/overlap_tiny'

    # The URL to get the data from (if it is not found locally)
    DATA_URL = 'http://nowhere'
    # Whether the testing computer has internet or not
    # (overrides all internet-based actions if set
    # to False)
    INTERNET = True
    # Whether to allow the program to download
    # the dataset if it's missing locally
    ALLOW_DOWNLOADS = False

    # The size of each image in the dataset.
    IMAGE_SIZE = 200

    "Training info_____________________________________________________________________________________________________"

    # Number of examples to load into the training queue
    NUM_EXAMPLES_TO_LOAD_INTO_QUEUE = 50

    # Number of examples per epoch for training and evaluation
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 9600
    NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 4800

    MAX_EPOCHS = 10
    TRAINING_STEP = 10000

    MAX_TRAIN_ITERATIONS = 10000

    ROTATE = True
    NUM_LAYERS = 3 if ROTATE else 2

    # Size of the training batch
    BATCH_SIZE = 100

    USE_FP16 = False

    TOWER_NAME = 'tower'

    NUM_THREADS = 2#55

    NOTIFICATION_EMAIL = 'andrew2000g+affinity@gmail.com'
    EMAIL_INFO = False

    RESTORE = False
    RESTORE_FROM = '../../../test_data/summaries/netstate/saved_state-91000'
    CHECKPOINT_DIR = './summaries'

    LEARNING_RATE = 1e-4
    CHECK_DATASET = True

    TRAIN_DIR = './train'
    SUMMARIES_DIR = './summaries'

    USE_TENSORBOARD = False

    KEEP_PROBABILITY = 0.5

    RUN_NAME = 'test'
    PRINT_INFO = True

    SEND_LOSS_TO_POOR_MANS_TENSORBOARD_EVERY_N_STEPS = 50
    SAVE_NETSTATE_AND_EMAIL_STATUS_EVERY_N_STEPS = 1000
