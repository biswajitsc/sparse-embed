import tensorflow as tf
from mobilenet import mobilenet_v2
import input_pipeline
import pprint
import time
from PIL import Image
import imagenet_preprocessing
import absl
from argparse import Namespace
import sys
import os
import numpy as np
import horovod.tensorflow as hvd
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow import logging
import mobilefacenet
import horovod.tensorflow as hvd
from mobilenet.mobilenet import global_pool
from mobilenet import conv_blocks as ops
from inception import inception_v1, inception_v2, inception_v3, inception_v4
from inception import inception_utils

inception_v1.default_image_size = 112
inception_v2.default_image_size = 112
inception_v3.default_image_size = 112
inception_v4.default_image_size = 112

slim = tf.contrib.slim
# Define all necessary hyperparameters as flags
flags = absl.flags
FLAGS = flags.FLAGS

expand_input = ops.expand_input_by_factor


# class ProxyInitializerHook(tf.train.SessionRunHook):
#   def begin(self):
#     print("*" * 20, "begin called")
#     with tf.variable_scope('Logits', reuse=True):
#       weights = tf.get_variable(name='proxy_wts')
#     with tf.variable_scope('proxy_init'):
#       weights_placeholder = tf.placeholder(dtype=tf.float32, name='weights_placeholder')
#       self.assign_op = tf.assign(weights, weights_placeholder)
  
#   def after_create_session(self, session, coord):
#     global_step = tf.train.get_or_create_global_step()
#     step = sess.run([global_step])
#     if step == 0:



def proxy_layer(embedding, num_classes):
  with tf.variable_scope('Logits'):
    weights = tf.get_variable(name='proxy_wts',
                              shape=(embedding.shape[-1], num_classes),
                              initializer=tf.random_normal_initializer(stddev=0.001),
                              dtype=tf.float32)
    weights = tf.nn.l2_normalize(weights, axis=0)
    # bias = tf.Variable(tf.zeros([num_classes]))
    alpha = 16.0
    logits = alpha * tf.matmul(embedding, weights) # + bias
  return logits


def arc_logits(embedding, out_num):
    with tf.variable_scope('Logits'):
      weights = tf.get_variable(name='proxy_wts', shape=(embedding.shape[-1], out_num),
                                initializer=tf.random_normal_initializer(stddev=0.001),
                                dtype=tf.float32)
      weights = tf.nn.l2_normalize(weights, axis=0)
    #   assign_op = tf.assign(weights, tf.nn.l2_normalize(weights, axis=0))
    #   with tf.control_dependencies([assign_op]):
      cos_t = tf.matmul(embedding, weights, name='cos_t')

    return cos_t


def soft_thresh(lam):
  # lam = tf.get_variable(
  #   "soft_thresh", shape=(),
  #   initializer=tf.zeros_initializer,
  #   constraint=lambda x: "hello")
  # lam = lam
  def activation(x):
    ret = tf.nn.relu(x - lam) - tf.nn.relu(-x - lam)
    return ret  
  return activation


def metric_learning(features, is_training, num_classes):
  # embedding, end_points = inception_v1.inception_v1(
  #   features, embedding_dim=FLAGS.embedding_size, use_bn=True, is_training=is_training,
  #   global_pool=True)
  embedding, end_points = inception_v4.inception_v4(
    features, num_classes=FLAGS.embedding_size, is_training=is_training,
    create_aux_logits=False)
  if FLAGS.final_activation == 'soft_thresh':
    act = soft_thresh(0.5)
  elif FLAGS.final_activation == 'relu':
    act = tf.nn.relu
  elif FLAGS.final_activation == 'none':
    act = lambda x: x
  else:
    raise ValueError('Unsupported activation', FLAGS.final_activation)
  embedding = act(embedding)
  embedding = tf.nn.l2_normalize(embedding, axis=1)
  end_points["embedding"] = embedding
  # net = tf.squeeze(net, axis=[1, 2])
  logits = proxy_layer(embedding, num_classes)
  end_points["Logits"] = logits

  return logits, end_points
