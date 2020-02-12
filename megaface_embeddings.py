from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import model
import pprint
import time
from PIL import Image
import absl
import sys
import os
import numpy as np
import horovod.tensorflow as hvd
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow import logging
import shutil
import glob
from sklearn.model_selection import KFold
import traceback
import imagenet_preprocessing
import pickle
import matio
import json
import multiprocessing
import tqdm


flags = absl.flags
flags.DEFINE_float(name='zero_threshold',
                   help='Threshold below which values will be considered zero',
                   default=1e-8)

flags.DEFINE_integer(name='batch_size', help='Batch size', default=64)
flags.DEFINE_integer(name='num_classes', help='Number of classes', default=9001)
flags.DEFINE_integer(name='num_epochs', help='Number of epochs', default=1)
flags.DEFINE_integer(name='embedding_size',
                     help='Embedding size',
                     default=1024)
flags.DEFINE_integer(name='num_threads_preprocess',
                     help='Number of threads for tensorflow preprocess pipeline',
                     default=50)
flags.DEFINE_integer(name='num_threads_save',
                     help='Number of threads for saving mat files',
                     default=50)
flags.DEFINE_integer(name='step',
                     help='The global step of the checkpoint to load',
                     default=None)

flags.DEFINE_string(name='model_dir',
                    help='Path to the model directory for checkpoints and summaries',
                    default='checkpoints/kubernetes/sparse_imagenet10k_dynamic_2_mom/')
flags.DEFINE_string(name='data_dir',
                    help='Directory containing the validation data',
                    default='../imagenet10k/tfrecords/val')
flags.DEFINE_string(name='filelist',
                    help='JSON file containing the list of files',
                    default=None)
flags.DEFINE_string(name='model',
                    help='Options are mobilenet and mobilefacenet',
                    default='mobilenet')
flags.DEFINE_string(name='output_suffix',
                    help='The output filename',
                    default='embeddings')
flags.DEFINE_string(name='final_activation',
                    help='The activation to use in the final layer producing the embeddings',
                    default=None)

flags.DEFINE_boolean(name='debug',
                     help='If True then inference will be stopped at 10000 steps',
                     default=True)
flags.DEFINE_boolean(name='is_training',
                     help='If True then embeddings for the training data will be computed',
                     default=False)

DENSE_DIR = "checkpoints/mobilenet_retrained/"
FLAGS = flags.FLAGS

def input_fn(filelist):
  dummy_labels = tf.constant([1] * len(filelist))
  filelist = tf.constant(filelist)
  dataset = tf.data.Dataset.from_tensor_slices((filelist, dummy_labels))

  def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.resize_images(image, [112, 112])
    image = tf.cast(image, dtype=tf.float32)
    image = image / 128.0 - 1.0
    return image, filename, label

  dataset = dataset.map(_parse_function,
                        num_parallel_calls=FLAGS.num_threads_preprocess)
  dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
  dataset = dataset.batch(FLAGS.batch_size)
  dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

  iterator = dataset.make_one_shot_iterator()
  image, filename, label = iterator.get_next()
  feature_dict = {
    "features": image,
    "labels": label,
    "label_texts": label,
    "filename": filename
  }

  return feature_dict, None


def get_global_step(ckpt_path=None):
  if ckpt_path is None:
    ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
  base = os.path.basename(ckpt_path)
  global_step = int(base.split('-')[1])
  return global_step


def main(_):
  filelist = json.load(open(FLAGS.filelist, 'r'))["path"]
  filelist = [os.path.join(FLAGS.data_dir, f) for f in filelist]

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config = tf.estimator.RunConfig(
    model_dir=FLAGS.model_dir,
    session_config=config
  )

  estimator = tf.estimator.Estimator(
    model_fn=model.model_fn,
    config=config
  )

  predictions = estimator.predict(
    input_fn=lambda: input_fn(filelist),
    yield_single_examples=True,
    predict_keys=["small_embeddings", "filename"],
    checkpoint_path=\
      tf.train.latest_checkpoint(FLAGS.model_dir)
        if FLAGS.step is None
        else 'model.ckpt-%d' % FLAGS.step
  )

  P = multiprocessing.Pool(FLAGS.num_threads_save)
  step = 0
  avg = 0.0

  for prediction in tqdm.tqdm(predictions):
    embedding = prediction["small_embeddings"]
    filename = prediction["filename"]
    filename = filename.decode() + FLAGS.output_suffix
    P.apply_async(matio.save_mat, (filename, embedding))
    step += 1
    avg = (1 - 1/step) * avg + np.sum(np.abs(embedding) >= 1e-8) / step
    if step % 10000 == 0:
      print('Step %d\tSparsity %f' % (step, avg))
  
  print('Step %d\tSparsity %f' % (step, avg))
  
  P.close()
  P.join()


if __name__=='__main__':
  absl.app.run(main)