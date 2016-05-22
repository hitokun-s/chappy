#!/usr/bin/env python
# -*- coding: utf-8 -*-

from unittest import TestCase
from nose.tools import ok_, eq_
from util import *
import os
import logging

class UtilTestCase(TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.DEBUG) # default is warning
        logging.debug("before test")

    def tearDown(self):
        logging.debug("after test")

    def create_mean_with_default_save_path_test(self):
        mean = create_mean(["resource/apple1.jpg","resource/apple2.jpg"])
        ok_(os.path.exists("mean.npy"))
        npy_to_img(mean[0],save_path="test1.png")
        npy_to_img(mean[1],save_path="test2.png")
        npy_to_img(mean[2],save_path="test3.png")

    def create_mean_with_save_path_test(self):
        create_mean(["resource/apple1.jpg","resource/apple2.jpg"], "new_file.npy")
        ok_(os.path.exists("new_file.npy"))

    def modify_test(self):
        eq_(modify("resource", ext="xls"), 0)
        eq_(modify("resource", ext="jpg"), 21)
        eq_(modify("resource", ext="png"), 21)

    def modify_with_dist_dir_test(self):
        eq_(modify("resource", ext="png", resize=(20,20), grayscale=True, dist_dir="resource/tmp"), 21)

    def modify_with_crop_test(self):
        modify("resource/raw", resize=(150,150),crop=True,dist_dir="resource/tmp")

    def npy_to_img_test(self):
        x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
        npy_to_img(x, save_path="test.png")
        npy_to_img(x, save_path="test.jpg")

    def img_to_npy_double_check_test(self):
        img_path = "resource/raw/apple1.jpg"
        arr1 = img_to_npy(img_path)
        arr2 = pil_to_npy(img_to_pil(img_path))
        ok_(np.array_equal(arr1, arr2))




