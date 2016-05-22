#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import datetime
import json
import multiprocessing
import os
import random
import sys
import threading
import time

import numpy as np
from PIL import Image
import six
import six.moves.cPickle as pickle
from six.moves import queue

import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers

# from lib.util import *

import mltl

testPath = "C:/Users/jgpua_000/ml/python-image-playground/data/train/non-face/B1_00308.pgm"
print(numpy.asarray(Image.open(testPath)).shape)
image = numpy.asarray(Image.open(testPath)).transpose(2, 0, 1)

def load_list(dirPath, classIdx):
    tuples = []
    cd = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    for filename in os.listdir(dirPath):
        tuples.append((cd + "/" + dirPath + "/" +  filename, np.int32(classIdx)))
    return tuples

# Prepare dataset
train_list = load_list("data/train/face", 0) + load_list("data/train/non-face", 1)
test_list = load_list("data/test/face", 0) + load_list("data/test/non-face", 1)

print(len(train_list))
# mean_image = pickle.load(open(args.mean, 'rb'))

mltl.create_mean([x[0] for x in train_list], "mean.npy")