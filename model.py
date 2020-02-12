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
import proxy_metric_learning
from inception import inception_utils
from inception import inception_v1
import resnet
from cifar import cifar
from cifar.dist import pairwise_distance_euclid, triplet_semihard_loss

slim = tf.contrib.slim
# Define all necessary hyperparameters as flags
flags = absl.flags
FLAGS = flags.FLAGS


def imagenet_iterator(is_training, num_epochs=1):
  filter_fn = None
  fixed_crop = False
  if 'face' in FLAGS.model:
    input_pipeline._DEFAULT_IMAGE_SIZE = 112
    imagenet_preprocessing._RESIZE_MIN = 112
    # imagenet_preprocessing._AREA_RANGE = [1.0, 1.0]
    # imagenet_preprocessing._ASPECT_RATIO_RANGE = [0.9, 1.1]
    # is_training = False
    fixed_crop = True

  if FLAGS.model == 'metric_learning':
    input_pipeline._DEFAULT_IMAGE_SIZE = 112
    imagenet_preprocessing._RESIZE_MIN = 112 
    imagenet_preprocessing._AREA_RANGE = [0.7, 1.0]
    imagenet_preprocessing._ASPECT_RATIO_RANGE = [0.9, 1.1]
    fixed_crop = False

  if FLAGS.model == 'mobilenet':
    imagenet_preprocessing._AREA_RANGE = [0.2, 1.0]
    fixed_crop = False

  if FLAGS.model == 'cifar100':
    return cifar.input_fn(FLAGS.data_dir, is_training, FLAGS.batch_size, num_epochs)
  else:
    data = input_pipeline.input_fn(
      is_training=is_training,
      data_dir=FLAGS.data_dir,
      batch_size=FLAGS.batch_size,
      num_epochs=num_epochs,
      return_label_text=True,
      filter_fn=filter_fn,
      fixed_crop=fixed_crop
    )

  iterator = data.make_one_shot_iterator()
  features, labels, label_texts, filename = iterator.get_next()
  feature_dict = {
    "features": features,
    "labels": labels,
    "label_texts": label_texts,
    "filename": filename
  }

  return feature_dict, labels


def model_fn(features, labels, mode):
  feature_dict = features
  labels = tf.reshape(feature_dict["labels"], [-1])

  if mode == tf.estimator.ModeKeys.TRAIN:
    is_training = True
  else:
    is_training = False

  if FLAGS.model == 'mobilenet':
    scope = mobilenet_v2.training_scope(is_training=is_training)
    with tf.contrib.slim.arg_scope(scope):
      net, end_points = mobilenet_v2.mobilenet(
        feature_dict["features"], is_training=is_training, num_classes=FLAGS.num_classes)
      end_points['embedding'] = end_points['global_pool']
  elif FLAGS.model == 'mobilefacenet':
    scope = mobilenet_v2.training_scope(is_training=is_training)
    with tf.contrib.slim.arg_scope(scope):
      net, end_points = mobilefacenet.mobilefacenet(
        feature_dict["features"], is_training=is_training, num_classes=FLAGS.num_classes)
  elif FLAGS.model == 'metric_learning':
    with slim.arg_scope(inception_v1.inception_v1_arg_scope(weight_decay=0.0)):
      net, end_points = proxy_metric_learning.metric_learning(
        feature_dict["features"], is_training=is_training, num_classes=FLAGS.num_classes)
  elif FLAGS.model == 'faceresnet':
    net, end_points = mobilefacenet.faceresnet(
      feature_dict["features"], is_training=is_training, num_classes=FLAGS.num_classes)
  elif FLAGS.model == 'cifar100':
    net, end_points = cifar.nin(feature_dict["features"], labels,
        is_training=is_training, num_classes=FLAGS.num_classes)
  else:
    raise ValueError("Unknown model %s" % FLAGS.model)

  small_embeddings = tf.squeeze(end_points['embedding'])
  logits = end_points['Logits']
  predictions = tf.cast(tf.argmax(logits, 1), dtype=tf.int32)

  if FLAGS.final_activation in ['soft_thresh', 'none']:
    abs_embeddings = tf.abs(small_embeddings)
  else:
    abs_embeddings = small_embeddings
  nnz_small_embeddings = tf.cast(tf.less(FLAGS.zero_threshold, abs_embeddings), dtype=tf.float32)
  small_sparsity = tf.reduce_sum(nnz_small_embeddings, axis=1)
  small_sparsity = tf.reduce_mean(small_sparsity)

  mean_nnz_col = tf.reduce_mean(nnz_small_embeddings, axis=0)
  sum_nnz_col = tf.reduce_sum(nnz_small_embeddings, axis=0)
  mean_flops_ub = tf.reduce_sum(sum_nnz_col * (sum_nnz_col - 1)) / (FLAGS.batch_size * (FLAGS.batch_size - 1))
  mean_flops = tf.reduce_sum(mean_nnz_col * mean_nnz_col)

  l1_norm_row = tf.reduce_sum(abs_embeddings, axis=1)
  l1_norm_col = tf.reduce_mean(abs_embeddings, axis=0)

  mean_l1_norm = tf.reduce_mean(l1_norm_row)
  mean_flops_sur = tf.reduce_sum(l1_norm_col * l1_norm_col)

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions_dict = {
      'predictions': predictions,
      'true_labels': feature_dict["labels"],
      'true_label_texts': feature_dict["label_texts"],
      'small_embeddings': small_embeddings,
      'sparsity/small': small_sparsity,
      'sparsity/flops': mean_flops_ub,
      'filename': features["filename"]
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions_dict)

  if FLAGS.model == 'mobilenet':
    cr_ent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
  elif 'face' in FLAGS.model:
    cr_ent_loss = mobilefacenet.arcface_loss(logits=logits, labels=labels, out_num=FLAGS.num_classes)
  elif FLAGS.model == 'metric_learning':
    cr_ent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    # cr_ent_loss = mobilefacenet.arcface_loss(logits=logits, labels=labels, out_num=FLAGS.num_classes)
  elif FLAGS.model == 'cifar100':
    # cr_ent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cr_ent_loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(
        labels, small_embeddings, margin=0.1)
    # cr_ent_loss = mobilefacenet.arcface_loss(logits=logits,
    #     labels=labels, out_num=FLAGS.num_classes, m=0.0)
    # cr_ent_loss = triplet_semihard_loss(labels=labels, embeddings=small_embeddings,
    #     pairwise_distance=lambda embed: pairwise_distance_euclid(embed, squared=True), margin=0.3)
  else:
    raise ValueError('Unknown model %s' % FLAGS.model)
  cr_ent_loss = tf.reduce_mean(cr_ent_loss) + tf.losses.get_regularization_loss()

  ema = tf.train.ExponentialMovingAverage(decay=0.99)
  ema_op = ema.apply([mean_l1_norm])
  moving_l1_norm = ema.average(mean_l1_norm)

  global_step = tf.train.get_or_create_global_step()
  all_ops = [ema_op]

  l1_weight = tf.Variable(0.0, name='l1_weight', trainable=False)

  if FLAGS.l1_weighing_scheme == 'constant':
    l1_weight = FLAGS.l1_parameter
  elif FLAGS.l1_weighing_scheme == 'dynamic_1':
    l1_weight = FLAGS.l1_parameter / moving_l1_norm
    l1_weight = tf.stop_gradient(l1_weight)
    l1_weight = tf.train.piecewise_constant(
      x=global_step, boundaries=[5], values=[0.0, l1_weight])
  elif FLAGS.l1_weighing_scheme == 'dynamic_2':
    if FLAGS.sparsity_type == "flops_sur":
      update_lr = 1e-5
    else:
      update_lr = 1e-4
    update = update_lr * (FLAGS.l1_parameter - cr_ent_loss)
    assign_op = tf.assign(l1_weight, tf.nn.relu(l1_weight + update))
    all_ops.append(assign_op)
  elif FLAGS.l1_weighing_scheme == 'dynamic_3':
    update_lr = 1e-4
    global_step = tf.train.get_or_create_global_step()
    upper_bound = FLAGS.l1_parameter - (FLAGS.l1_parameter - 12.0) * tf.cast(global_step, tf.float32) / 5e5
    upper_bound = tf.cast(upper_bound, tf.float32)
    update = update_lr * tf.sign(upper_bound - cr_ent_loss)
    assign_op = tf.assign(l1_weight, l1_weight + tf.nn.relu(update))
    all_ops.append(assign_op)
  elif FLAGS.l1_weighing_scheme == 'dynamic_4':
    l1_weight = FLAGS.l1_parameter * tf.minimum(1.0,
        (tf.cast(global_step, tf.float32) / FLAGS.l1_p_steps)**2)
  elif FLAGS.l1_weighing_scheme is None:
    l1_weight = 0.0
  else:
    raise ValueError('Unknown l1_weighing_scheme %s' % FLAGS.l1_weighing_scheme)

  if FLAGS.sparsity_type == 'l1_norm':
    sparsity_loss = l1_weight * mean_l1_norm
  elif FLAGS.sparsity_type == 'flops_sur':
    sparsity_loss = l1_weight * mean_flops_sur
  elif FLAGS.sparsity_type is None:
    sparsity_loss = 0.0
  else:
    raise ValueError("Unknown sparsity_type %d" % FLAGS.sparsity_type)

  total_loss = cr_ent_loss + sparsity_loss
  accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), dtype=tf.float32))

  if mode == tf.estimator.ModeKeys.EVAL:
    global_accuracy = tf.metrics.mean(accuracy)
    global_small_sparsity = tf.metrics.mean(small_sparsity)
    global_flops = tf.metrics.mean(mean_flops_ub)
    global_l1_norm = tf.metrics.mean(mean_l1_norm)
    global_mean_flops_sur = tf.metrics.mean(mean_flops_sur)
    global_cr_ent_loss = tf.metrics.mean(cr_ent_loss)

    metrics = {
      'accuracy': global_accuracy,
      'sparsity/small': global_small_sparsity,
      'sparsity/flops': global_flops,
      'l1_norm': global_l1_norm,
      'l1_norm/flops_sur': global_mean_flops_sur,
      'loss/cr_ent_loss': global_cr_ent_loss,
    }
    return tf.estimator.EstimatorSpec(mode, loss=total_loss, eval_metric_ops=metrics)

  if mode == tf.estimator.ModeKeys.TRAIN:
    base_lrate = FLAGS.learning_rate
    learning_rate = tf.train.piecewise_constant(
      x=global_step,
      boundaries=[FLAGS.decay_step if FLAGS.decay_step is not None else int(1e7)],
      values=[base_lrate, base_lrate / 10.0])

    tf.summary.image("input", feature_dict["features"], max_outputs=1)
    tf.summary.scalar('sparsity/small', small_sparsity)
    tf.summary.scalar('sparsity/flops', mean_flops_ub)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('l1_norm', mean_l1_norm)
    tf.summary.scalar('l1_norm/ema', moving_l1_norm)  # Comment this for mom
    tf.summary.scalar('l1_norm/l1_weight', l1_weight)
    tf.summary.scalar('l1_norm/flops_sur', mean_flops_sur)
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('loss/cr_ent_loss', cr_ent_loss)
    tf.summary.scalar('sparsity/ratio',
                      mean_flops * FLAGS.embedding_size / (small_sparsity * small_sparsity))
    try:
      tf.summary.scalar('loss/upper_bound', upper_bound)
    except NameError:
      print("Skipping 'upper_bound' summary")
    for variable in tf.trainable_variables():
      if 'soft_thresh' in variable.name:
        print('Adding summary for lambda')
        tf.summary.scalar('lambda', variable)
    
    # Histogram summaries
    # gamma = tf.get_default_graph().get_tensor_by_name('MobilenetV2/Conv_2/BatchNorm/gamma:0')
    # pre_relu = tf.get_default_graph().get_tensor_by_name(
    #       'MobilenetV2/Conv_2/BatchNorm/FusedBatchNorm:0')
    # pre_relu = tf.squeeze(pre_relu)
    # tf.summary.histogram('gamma', gamma)
    # tf.summary.histogram('pre_relu', pre_relu[:, 237])
    # tf.summary.histogram('small_activations', nnz_small_embeddings)
    # tf.summary.histogram('small_activations/log', tf.log(nnz_small_embeddings + 1e-10))
    # fl_sur_ratio = (mean_nnz_col * mean_nnz_col) / (l1_norm_col * l1_norm_col)
    # tf.summary.histogram('fl_sur_ratio', fl_sur_ratio)
    # tf.summary.histogram('fl_sur_ratio/log', tf.log(fl_sur_ratio + 1e-10))
    # l1_sur_ratio = (mean_nnz_col * mean_nnz_col) / l1_norm_col
    # tf.summary.histogram('l1_sur_ratio', l1_sur_ratio)
    # tf.summary.histogram('l1_sur_ratio/log', tf.log(l1_sur_ratio + 1e-10))

    if FLAGS.optimizer == 'mom':
      optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                             momentum=FLAGS.momentum)
    elif FLAGS.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif FLAGS.optimizer == 'adam':
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.0001)
    elif FLAGS.optimizer == 'rmsprop':
      optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, epsilon=0.01,
        momentum=FLAGS.momentum)
    else:
      raise ValueError("Unknown optimizer %s" % FLAGS.optimizer)

    # Make the optimizer distributed
    optimizer = hvd.DistributedOptimizer(optimizer)

    train_op = tf.contrib.slim.learning.create_train_op(
      total_loss,
      optimizer,
      global_step=tf.train.get_or_create_global_step()
    )
    all_ops.append(train_op)
    merged = tf.group(*all_ops)

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=merged)
