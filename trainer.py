#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys

import numpy as np
from PIL import Image
import scipy.misc
from matplotlib import pyplot as plt
import six.moves.cPickle as pickle
import six

import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers

from logging import getLogger

logger = getLogger(__name__)

using_gpu = False
xp = np
try:
    cuda.check_cuda_available()
    xp = cuda.cupy
    cuda.get_device(0).use()
    using_gpu = True
except:
    logger.info("I'm sorry. Using CPU.")

class Trainer():

    def __init__(self, train_data, optimizer, model, test_data=None, n_epoch=100, batch_size=100, save_interval=10, save_path=""):

        # train_d, test_dataは２通り想定。
        # 1. [(<data1>, classIdx1), (<data2>, classIdx2),,,,] というリスト
        # 2. ([<data1>,<data2>,,,,], [classidx1, classIdx2,,,,])というタプル
        # これらは転置の関係にある。

        if isinstance(train_data, tuple):
            self.x_train, self.y_train = train_data # array of tuple as (<data>, classIdx) <= data is either normal array, numpy array
            self.x_test, self.y_test = test_data # array of tuple as (<data>, classIdx) <= data is either normal array, numpy array
        else:
            self.x_train, self.y_train = np.asarray(train_data).transpose()
            self.x_test, self.y_test = np.asarray(test_data).transpose()

        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.model = model
        self.save_interval = save_interval

        self.optimizer.setup(model)

        if os.path.exists(save_path):
            os.mkdir(save_path)

    def __save(self):
        serializers.save_npz(self.save_path, self.model)
        serializers.save_npz(self.save_path, self.optimizer)

    def train(self):
        N = len(self.x_train)
        N_test = len(self.x_test)
        for epoch in six.moves.range(1, self.n_epoch + 1):
            print('epoch', epoch)

            # training
            perm = np.random.permutation(N) # ランダム数列を作る（例：permutation(5) => [3,1,4,0,2]）
            sum_accuracy = 0
            sum_loss = 0
            range = six.moves.range(0, N, self.batch_size)
            print(range)
            for i in range:
                x = chainer.Variable(xp.asarray(self.x_train[perm[i:i + self.batch_size]]))
                t = chainer.Variable(xp.asarray(self.y_train[perm[i:i + self.batch_size]]))

                # Pass the loss function (Classifier defines it) and its arguments
                self.optimizer.update(self.model, x, t)

                # if epoch == 1 and i == 0:
                #     with open('graph.dot', 'w') as o:
                #         g = computational_graph.build_computational_graph((self.model.loss,), remove_split=True)
                #         o.write(g.dump())
                #     print('graph generated')

                sum_loss += float(self.model.loss.data) * len(t.data)
                sum_accuracy += float(self.model.accuracy.data) * len(t.data)

            print('train mean loss={}, accuracy={}'.format( sum_loss / N, sum_accuracy / N))

            # evaluation
            sum_accuracy = 0
            sum_loss = 0
            for i in six.moves.range(0, N_test, self.batch_size):
                x = chainer.Variable(xp.asarray(self.x_test[i:i + self.batch_size]),volatile='on')
                t = chainer.Variable(xp.asarray(self.y_test[i:i + self.batch_size]),volatile='on')
                loss = self.model(x, t)
                # print self.model.predictor(x).data # 最終的なネットワーク出力（クラス数分の小数の配列）

                sum_loss += float(loss.data) * len(t.data)
                sum_accuracy += float(self.model.accuracy.data) * len(t.data)

            print('test  mean loss={}, accuracy={}'.format(sum_loss / N_test, sum_accuracy / N_test))

            if (epoch + 1) % self.save_interval == 0:
                self.__save()

        self.__save()

