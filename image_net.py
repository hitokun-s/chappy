#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys

import numpy as np
from PIL import Image
import scipy.misc
from matplotlib import pyplot as plt
import six.moves.cPickle as pickle
import xml.etree.ElementTree as ET
from logging import getLogger
logger = getLogger(__name__)

API_BASE = "http://www.image-net.org/api"

# 全ての子孫synsetを取得
API_ALL_SUB_SYNSET = API_BASE + "/text/wordnet.structure.hyponym?full=1&wnid=%(wnid)s"

# 画像IDと画像URLのセットのリストを取得
API_IMG_URLS_WITH_ID = API_BASE + "/text/imagenet.synset.geturls.getmapping?wnid=%(wnid)s"

# あるwnidに属する各画像のbbox情報（xml）をまとめたtar.gzファイルのダウンロードリンク
# API_BBOX_DOWNLOAD = API_BASE + "/download/imagenet.bbox.synset?wnid=%(wnid)s"
API_BBOX_DOWNLOAD = "http://image-net.org/downloads/bbox/bbox/%(wnid)s.tar.gz"

from urllib2 import urlopen
from urllib import urlretrieve
from bs4 import BeautifulSoup
import sqlite3
import tarfile
# from bbox_helper import *

def __get_lines_from_web(url):
    f = urlopen(url)
    soup = BeautifulSoup(f, "html.parser")
    return soup.text.strip().split("\n")

def __get_child_file_paths(dir_path):
    logger.debug(dir_path)
    return [os.path.join(dir_path, filename) for filename in os.listdir(dir_path)]

def get_all_sub_synset(wnid):
    lines = __get_lines_from_web(API_ALL_SUB_SYNSET % {"wnid":wnid})
    return [line[1:] for line in lines[1:]]

def get_img_urls(wnid):
    lines = __get_lines_from_web(API_IMG_URLS_WITH_ID % {"wnid":wnid})
    print "wnid:" + wnid +   ",count:%d" % len(lines)
    return [line.split() for line in lines] # タプル(image_id,url)のリストにする\

def download_bbox_xml(wnid, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    filename = "%s.tar.gz" % wnid
    save_path = os.path.join(save_dir, filename)
    # if os.path.exists(savePath):
    #     print "already exists:%s" % savePath
    #     return
    url = API_BBOX_DOWNLOAD % {"wnid":wnid}
    # if not isValidUrl(url):
    #     print "invalid url:%s" % url
    #     return
    urlretrieve(url, save_path)
    tf = tarfile.open(save_path, "r:gz")
    tf.extractall(save_dir)
    tf.close()
    os.remove(save_path)
    return __get_child_file_paths(os.path.join(save_dir, "Annotation", wnid))

def get_bbox(xml_path):
    logger.debug(xml_path)
    xmltree = ET.parse(xml_path)
    filename = xmltree.find('filename').text
    wnid, image_id = filename.split('_')
    objects = xmltree.findall('object')
    rects = []
    for object_iter in objects:
        bndbox = object_iter.find("bndbox")
        rects.append([int(it.text) for it in bndbox])
    return rects

def crop_bbox_(self, img_path, rects, save_dir=None):

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    bbs = []
    im = Image.open(self.img_path)
    for box in rects:
        bbs.append(im.crop(box))
    count = 0
    for box in bbs:
        count = count + 1
        outPath = str(os.path.join(save_dir, self.annotation_filename + '_box' + str(count) + '.JPEG'))
        box.save(outPath)
        print 'save to ' + outPath
