#!/usr/bin/env python
# -*- coding: utf-8 -*-

from unittest import TestCase
from nose.tools import ok_, eq_
from image_net import *
import os
import logging

class ImageNetTestCase(TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.DEBUG) # default is warning
        logging.debug("before test")

    def tearDown(self):
        logging.debug("after test")

    def get_all_sub_synset_test(self):
        res = get_all_sub_synset("n07707451") # 野菜
        logging.info(len(res))

    def get_img_urls_test(self):
        res = get_img_urls("n07707451")
        for v in res:
            logging.debug(v)

    def download_bbox_xml_test(self):
        file_paths = download_bbox_xml("n07707451", "resource/bbox")
        logging.info(get_bbox(file_paths[0]))

