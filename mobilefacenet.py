import tensorflow as tf
from mobilenet import mobilenet_v2
from mobilenet import conv_blocks as ops
from mobilenet import mobilenet as lib
import input_pipeline as input_pipeline
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
import math
from resnet import resnet_model

# Define all necessary hyperparameters as flags
flags = absl.flags
FLAGS = flags.FLAGS

slim = tf.contrib.slim
op = lib.op

expand_input = ops.expand_input_by_factor


# The Arcface loss function is taken from
# https://github.com/auroua/InsightFace_TF/blob/master/losses/face_losses.py
def arcface_logits(embedding, out_num):
    with tf.variable_scope('Arcface'):
      weights = tf.get_variable(name='embedding_weights', shape=(embedding.shape[-1], out_num),
                                initializer=tf.random_normal_initializer(stddev=0.001),
                                dtype=tf.float32)
      weights = tf.nn.l2_normalize(weights, axis=0)
    #   assign_op = tf.assign(weights, tf.nn.l2_normalize(weights, axis=0))
    #   with tf.control_dependencies([assign_op]):
      cos_t = tf.matmul(embedding, weights, name='cos_t')

    return cos_t


def arcface_loss(logits, labels, out_num, s=64.0, m=0.5):
  assert len(labels.shape) == 1
  cos_m = math.cos(m)
  sin_m = math.sin(m)
  threshold = math.cos(math.pi - m)

  mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask', on_value=1.0, off_value=0.0)
  cos_t = tf.multiply(logits, mask, name='cos_t')
  cos_t = tf.reduce_sum(cos_t, axis=1)

  cos_t2 = tf.square(cos_t, name='cos_t2')
  sin_t2 = tf.subtract(1.0, cos_t2, name='sin_t2')
  sin_t = tf.sqrt(sin_t2, name='sin_t')

  cos_mt = tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

  cond = tf.less(threshold, cos_t, name='if_else')
  keep_val = cos_t - (threshold + 1)
  cos_mt = tf.where(cond, cos_mt, keep_val)
  # cos_mt = cond * cos_mt + (1 - cond) * keep_val

  update = tf.reshape(cos_mt, [-1, 1]) * mask
  mask = tf.cast(mask, tf.bool)
  logits = s * tf.where(mask, update, logits)
  # logits = s * (tf.reshape(cos_mt, [-1, 1]) * mask + logits * (1.0 - mask))
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
  return loss


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


def sign(x):
  return tf.stop_gradient(tf.sign(x) - tf.clip_by_value(x, -1, 1)) \
      + tf.clip_by_value(x, -1, 1)


def step(x):
  return (sign(x) + 1) / 2


def mobilefacenet(features, is_training, num_classes):
  if FLAGS.final_activation == 'step':
    final_activation = step
  elif FLAGS.final_activation == 'relu':
    final_activation = tf.nn.relu
  elif FLAGS.final_activation == 'relu6':
    final_activation = tf.nn.relu6
  elif FLAGS.final_activation == 'soft_thresh':
    final_activation = soft_thresh(0.5)
  elif FLAGS.final_activation == 'none':
    final_activation = None
  else:
    raise ValueError('Unknown activation %s' % str(FLAGS.final_activation))

  MFACENET_DEF = dict(
    defaults={
      # Note: these parameters of batch norm affect the architecture
      # that's why they are here and not in training_scope.
      (slim.batch_norm,): {'center': True, 'scale': True},
      (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
          'normalizer_fn': slim.batch_norm, 'activation_fn': tf.nn.leaky_relu,
          'weights_regularizer': slim.l2_regularizer(4e-5)
      },
      (ops.expanded_conv,): {
          'split_expansion': 1,
          'normalizer_fn': slim.batch_norm,
          'residual': True
      },
      (slim.conv2d, slim.separable_conv2d): {'padding': 'SAME'},
    },
    spec=[
      op(slim.conv2d, stride=2, num_outputs=64, kernel_size=[3, 3]),

      op(slim.separable_conv2d, num_outputs=None, kernel_size=[3, 3],
          depth_multiplier=1, stride=1),

      op(ops.expanded_conv, expansion_size=expand_input(2), stride=2, num_outputs=64),
      op(ops.expanded_conv, expansion_size=expand_input(2), stride=1, num_outputs=64),
      op(ops.expanded_conv, expansion_size=expand_input(2), stride=1, num_outputs=64),
      op(ops.expanded_conv, expansion_size=expand_input(2), stride=1, num_outputs=64),
      op(ops.expanded_conv, expansion_size=expand_input(2), stride=1, num_outputs=64),

      op(ops.expanded_conv, expansion_size=expand_input(4), stride=2, num_outputs=128),

      op(ops.expanded_conv, expansion_size=expand_input(2), stride=1, num_outputs=128),
      op(ops.expanded_conv, expansion_size=expand_input(2), stride=1, num_outputs=128),
      op(ops.expanded_conv, expansion_size=expand_input(2), stride=1, num_outputs=128),
      op(ops.expanded_conv, expansion_size=expand_input(2), stride=1, num_outputs=128),
      op(ops.expanded_conv, expansion_size=expand_input(2), stride=1, num_outputs=128),
      op(ops.expanded_conv, expansion_size=expand_input(2), stride=1, num_outputs=128),

      op(ops.expanded_conv, expansion_size=expand_input(4), stride=2, num_outputs=128),

      op(ops.expanded_conv, expansion_size=expand_input(2), stride=1, num_outputs=128),
      op(ops.expanded_conv, expansion_size=expand_input(2), stride=1, num_outputs=128),

      op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=512),

      op(slim.separable_conv2d, num_outputs=None, kernel_size=[7, 7],
         depth_multiplier=1, stride=1, padding="VALID", activation_fn=None),

      op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=FLAGS.embedding_size,
         weights_regularizer=slim.l2_regularizer(4e-4), activation_fn=final_activation),
    ],
  )

  net, end_points = mobilenet_v2.mobilenet(
      features, is_training=is_training, num_classes=num_classes,
      base_only=True, conv_defs=MFACENET_DEF)

  embedding = end_points['embedding']
  embedding = tf.squeeze(embedding, [1, 2])
  embedding_unnorm = embedding
  embedding = tf.nn.l2_normalize(embedding, axis=1)
  if FLAGS.final_activation == 'step':
    end_points['embedding'] = embedding_unnorm
  else:
    end_points['embedding'] = embedding
  logits = arcface_logits(embedding, num_classes)
  end_points['Logits'] = logits

  return logits, end_points


def faceresnet(features, is_training, num_classes):
  if FLAGS.final_activation == 'step':
    final_activation = step
  elif FLAGS.final_activation == 'relu':
    final_activation = tf.nn.relu
  elif FLAGS.final_activation == 'relu6':
    final_activation = tf.nn.relu6
  elif FLAGS.final_activation == 'soft_thresh':
    final_activation = soft_thresh(0.5)
  elif FLAGS.final_activation == 'none':
    final_activation = None

  filter_list = [64, 64, 128, 256, 512]
  units = [3, 13, 30, 3]

  model = resnet_model.Model(
    resnet_size=100, bottleneck=False, num_classes=FLAGS.embedding_size,
    num_filters=filter_list, kernel_size=(3, 3), conv_stride=1,
    first_pool_size=None, first_pool_stride=None,
    block_sizes=units, block_strides=[2] * 4, resnet_version=3,
    data_format='channels_last')
  # Relu/soft_thresh is not applied in the model
  embedding = model(features, is_training)
  if final_activation is not None:
    embedding = final_activation(embedding)
  embedding = tf.nn.l2_normalize(embedding, axis=1)

  end_points = dict()
  end_points['embedding'] = embedding
  logits = arcface_logits(embedding, num_classes)
  end_points['Logits'] = logits

  return logits, end_points
