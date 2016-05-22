#!/usr/bin/env python
# -*- coding: utf-8 -*-

from unittest import TestCase
from nose.tools import ok_, eq_
import util
from trainer import *
import os
import logging

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers

import mnist_util.data as data
import mnist_util.net as net

class TrainerTestCase(TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.DEBUG) # default is warning
        logging.debug("before test")

    def tearDown(self):
        logging.debug("after test")

    def train_mnist_test(self):
        mnist = data.load_mnist_data() # 初回だけMNISTデータがロードされる
        mnist['data'] = mnist['data'].astype(np.float32)
        mnist['data'] /= 255
        mnist['target'] = mnist['target'].astype(np.int32)

        x_train, x_test = np.split(mnist['data'],   [60000])
        y_train, y_test = np.split(mnist['target'], [60000])

        n_units = 1000
        nn = net.MnistMLP(784, n_units, 10)
        model = L.Classifier(nn)

        if util.using_gpu():
            model.to_gpu()

        trainer = Trainer(
                model=model,
                optimizer=optimizers.Adam(),
                train_data=(x_train, y_train),
                test_data=(x_test, y_test)
        )
        trainer.train()

