#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys

import numpy as np
from PIL import Image
import scipy.misc
from matplotlib import pyplot as plt
import six.moves.cPickle as pickle
from logging import getLogger

from chainer import cuda

logger = getLogger(__name__)



def create_mean(filePaths, save_path="mean.npy", root_dir=None):
    """
    Create mean image as npy file.
    :param filePaths:list of source image file paths
    :param save_path:
    :param root_dir:
    :return: mean image data as numpy array
    """
    logger.info("start create_mean")

    sum_image = None
    count = 0
    for filepath in filePaths:
        # if not imgpath.endswith(".jpg") and not imgpath.endswith(".JPEG"):
        #     print "not jpg file. skip..."
        #     continue
        try:
            image = np.asarray(Image.open(filepath)).transpose(2, 0, 1)
        except:
            logger.debug("file:%s" % filepath)
            print sys.exc_info()[0]
            continue
        if image is None:
            continue
        if sum_image is None:
            sum_image = np.ndarray(image.shape, dtype=np.float32)
            sum_image[:] = image
        else:
            # ここで、operands could not be broadcast together with shapes エラーになった
            # broadcast ... サイズ/形状の異なる配列同士の演算のこと
            sum_image += image
        count += 1
    mean = sum_image / count
    pickle.dump(mean, open(save_path, 'wb'), -1)
    logger.info("mean image created!")
    return mean


def modify(source_dir, ext=None, resize=None, crop=False, format=None, dist_dir=None, grayscale=False):
    """
    modify file format and/or size, like ImageMagick
    :param image_dir:
    :param format:ex. jpg, png
    :param resize:[tupple] (width, hright)
    :param crop: crop as maximum square centered
    :param format: ex.BMP, GIF, JPEG, PNG (see https://infohost.nmt.edu/tcc/help/pubs/pil/formats.html)
    :return: processed count
    """
    # for i, imgpath in enumerate(os.listdir(image_dir)):
    #     filename, ext = os.path.splitext(imgpath)
    #     if not imgpath.endswith(".jpg") and not imgpath.endswith(".JPEG"):
    #         print "not image file!:%s" % imgpath
    #         continue

    if dist_dir is None:
        dist_dir = source_dir
    elif not os.path.exists(dist_dir):
        logger.info("create dist dir:%s" % dist_dir)
        os.mkdir(dist_dir)

    targets = [os.path.join(source_dir, path) for path in os.listdir(source_dir) if
               ext is None or os.path.splitext(path)[1][1:] == ext]
    for imgpath in targets:

        logger.debug("imgpath:%s" % imgpath)
        img = Image.open(imgpath)

        if crop:
            size = min(img.size)  # img.size は、(width, height)というタプルを返す。PILのバージョンによっては、img.width, img.heightも使えるが。
            start_x = img.size[0] / 2 - size / 2
            start_y = img.size[1] / 2 - size / 2
            box = (start_x, start_y, start_x + size,
                   start_y + size)  # box is a 4-tuple defining the left, upper, right, and lower pixel coordinate.
            img = img.crop(box)

        if resize is not None:
            img = img.resize(resize, Image.ANTIALIAS)

        if grayscale:
            img = img.convert('LA')

        basename = os.path.basename(imgpath)  # ex. hoge.jpg

        if format is not None:
            basename = chg_ext(basename, format.lower())

        save_path = os.path.join(dist_dir, basename)

        logger.debug("save img path:%s" % save_path)
        img.save(save_path, format)

    return len(targets)

# img : image file
# npy : numpy ndarray object
# pil : PIL object

def npy_to_img(img_array, file_path=None, save_path=None, format="PNG"):
    if img_array is None:
        img_array = np.load(file_path)
    logger.debug("shape:{0}".format(img_array.shape))
    if save_path is None:
        save_path = chg_ext(file_path, format.lower())
    plt.imshow(img_array)
    plt.savefig(save_path)

def img_to_npy(img_path):
    return pil_to_npy(Image.open(img_path))

def pil_to_npy(pil_img):
    imgArray = np.asarray(pil_img)
    imgArray.flags.writeable = True
    return imgArray

def npy_to_pil(img_arr):
    return Image.fromarray(np.uint8(img_arr))

def pil_to_img(pil_img, save_path):
    pil_img.save(save_path)

def img_to_pil(img_path):
    return Image.open(img_path)

def toGrayScale(pil_img):
    return pil_img.convert("L")

# ファイルパスの拡張子を変えて返す
def chg_ext(filename, ext):
    return filename.replace(os.path.split(filename)[1][1:], ext)

def using_gpu():
    try:
        cuda.check_cuda_available()
        # xp = cuda.cupy
        # cuda.get_device(0).use()
        return True
    except:
        return False
