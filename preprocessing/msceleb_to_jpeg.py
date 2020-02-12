'''
Extracts jpeg images from MXNet format dataset.
Usage msceleb_to_jpeg.py dataset_directory
Extracted images will be stored in dataset_directory/images
'''

import os
import random
import logging
import sys
import numbers
import numpy as np

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio
from PIL import Image
import pickle


def get_msceleb_images(records_dir):
  imgidx_path = os.path.join(records_dir, "train.idx")
  imgrec_path = os.path.join(records_dir, "train.rec")
  images_dir = os.path.join(records_dir, "images")

  imgrec = recordio.MXIndexedRecordIO(imgidx_path, imgrec_path, 'r')
  s = imgrec.read_idx(0)
  header, _ = recordio.unpack(s)
  tot_images = int(header.label[0]) - 1
  print("Total images", tot_images)
  for i in range(tot_images):
    print("Reading ", i)
    s = imgrec.read()
    header, img = recordio.unpack(s)
    img = mx.image.imdecode(img).asnumpy()
    label = int(header.label)
    img = Image.fromarray(np.uint8(img), "RGB")
    images_subdir = os.path.join(images_dir, "identity_%d" % label)
    if not os.path.exists(images_subdir):
      os.makedirs(images_subdir)
    image_path = os.path.join(images_subdir, "image_%d.jpg" % i)
    img.save(image_path)


def get_lfw_images(records_dir):
  lfw_bin = os.path.join(records_dir, "lfw.bin")
  bins, issame_list = pickle.load(open(lfw_bin, "rb"), encoding='bytes')
  lfw_images = os.path.join(records_dir, "lfw_images")
  if not os.path.exists(lfw_images):
    os.makedirs(lfw_images)

  for i in range(1000):
    bin_ = bins[i]
    img = mx.image.imdecode(bin_).asnumpy()
    img = Image.fromarray(np.uint8(img), "RGB")
    image_path = os.path.join(lfw_images, "image_%d.jpg" % i)
    img.save(image_path)


def main(records_dir):
  get_msceleb_images(records_dir)
  # get_lfw_images(records_dir)


if __name__ == "__main__":
  main(sys.argv[1])
