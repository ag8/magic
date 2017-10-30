#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.misc

import tensorflow as tf

import tensorflow.contrib.slim as slim

import overlap_input



images, labels = overlap_input.inputs(reshape=True)

print("Woo!")

