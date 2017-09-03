#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
from sys import stdout

import math
from time import sleep


def printl(text):
   print(text, end='')


def magic_print(text, epoch, max_epochs, i, max_i):
    rows, columns = os.popen('stty size', 'r').read().split()
    rows = int(rows)
    columns = int(columns)

    total_length = columns / 2

    # print("i=" + str(i))

    # print("Total length: " + str(total_length))

    bar_length_e = int(math.floor(float(total_length) * float(epoch) / float(max_epochs)))
    bar_length_i = int(float(total_length) * float((i + 1) * float(epoch + 1)) / float(max_i * max_epochs))

    bar_length = bar_length_e
    rest_length = total_length - bar_length

    # print("Bar length: " + str(bar_length))

    bar = '█' * bar_length
    rest = ' ' * rest_length
    bottom_line = 'Training...    |' + bar + '' + rest + '| ' + get_current_loader(i)


    stdout.write("\r" + " " * columns)
    stdout.flush()
    stdout.write("\r" + text)
    stdout.flush()
    stdout.write("\n")
    stdout.write(bottom_line)
    stdout.flush()

    sleep(1)


def get_current_loader(i):
    j = i / 5  # Evenly divisible

    loaders = ['/', '-', '\\', '-']

    return loaders[j % 4]
