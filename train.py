import tensorflow as tf
import model
import pprint
import time
import datetime
from PIL import Image
import absl
import sys
import os
import numpy as np
import horovod.tensorflow as hvd
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow import logging
import resource
import flags
import evaluate_retrieval
import evaluate_mfacenet
import traceback
from cifar import cifar


# Define all necessary hyperparameters as flags
FLAGS = absl.flags.FLAGS
IMAGENET_NUM_TRAINING = 1281167


class ImageRateHook(tf.train.StepCounterHook):
  '''
  A simple extension of StepCounterHook to count images/sec.
  Overriding the _log_and_record function is a simple hack to achieve this.
  '''
  def _log_and_record(self, elapsed_steps, elapsed_time, global_step):
    images_per_sec = elapsed_steps / elapsed_time * FLAGS.batch_size * hvd.size()
    summary_tag = 'images/sec'
    if self._summary_writer is not None:
      summary = Summary(value=[Summary.Value(
          tag=summary_tag, simple_value=images_per_sec)])
      self._summary_writer.add_summary(summary, global_step)
    logging.info("%s: %g", summary_tag, images_per_sec)


def main(_):
  hvd.init()

  # Only a see a single unique GPU based on process rank
  config = tf.ConfigProto()

  # We don't need allow_growth=True as we will use a whole GPU for each process.
  # config.gpu_options.allow_growth = True
  config.gpu_options.visible_device_list = str(hvd.local_rank())

  # Only one of the workers save checkpoints and summaries in the
  # model directory
  if hvd.rank() == 0:
    config = tf.estimator.RunConfig(
      model_dir=FLAGS.model_dir,
      keep_checkpoint_every_n_hours=5,
      save_summary_steps=100,
      save_checkpoints_secs=60 * 5,
      session_config=config
    )
  else:
    config = tf.estimator.RunConfig(
      session_config=config,
      keep_checkpoint_max=1
    )

  if FLAGS.mobilenet_checkpoint_path is not None:
    # ^((?!badword).)*$ matches all strings which do not contain the badword
    ws = tf.estimator.WarmStartSettings(
      ckpt_to_initialize_from=FLAGS.mobilenet_checkpoint_path,
      vars_to_warm_start='.*' if FLAGS.restore_last_layer else "^((?!Logits).)*$",
    )
  else:
    ws = None

  estimator = tf.estimator.Estimator(
    model_fn=model.model_fn,
    config=config,
    warm_start_from=ws
  )

  bcast_hook = hvd.BroadcastGlobalVariablesHook(0)
  image_counter_hook = ImageRateHook()
  # Caching file writers, which will then be retrieved during evaluation.
  writer = tf.summary.FileWriter(logdir=FLAGS.model_dir, flush_secs=30)
  eval_writer = tf.summary.FileWriter(
    logdir=os.path.join(FLAGS.model_dir, "eval"), flush_secs=30)

  try:
    steps = estimator.get_variable_value('global_step')
  except ValueError:
    steps = 0
  evaluate_every_n = 1000
  evaluate(estimator, True)
  evaluate(estimator, False)
  # if hvd.rank() == 0 and FLAGS.evaluate:
  #   evaluate(estimator, False)
  sys.exit()
  print("Steps", steps, "Max steps", FLAGS.max_steps)
  while steps < FLAGS.max_steps:
    evaluate_every_n = min(evaluate_every_n, FLAGS.max_steps - steps)
    estimator.train(input_fn=lambda: model.imagenet_iterator(
                      is_training=True, num_epochs=10000),
                    steps=evaluate_every_n,
                    hooks=[bcast_hook, image_counter_hook])
    if hvd.rank() == 0 and FLAGS.evaluate:
      # Evaluate on training set only for metric_learning
      if FLAGS.model in ['metric_learning', 'cifar100']:
        evaluate(estimator, True)
        evaluate(estimator, False)
      else:
        evaluate(estimator, False)
    
    steps += evaluate_every_n


def evaluate(estimator, is_training):
  if is_training:
    logdir = FLAGS.model_dir
  else:
    logdir = os.path.join(FLAGS.model_dir, 'eval')
  print('Writing to', logdir)
  print('Evaluating for %s' % FLAGS.model)
  writer = tf.summary.FileWriterCache.get(logdir=logdir)
  if FLAGS.model in ['metric_learning']:
    global_step = evaluate_retrieval.get_global_step()
    evaluate_retrieval.compute_metrics(estimator, global_step, is_training, writer)
  if FLAGS.model in ['cifar100']:
    global_step = evaluate_retrieval.get_global_step()
    cifar.compute_metrics(estimator, global_step, is_training, writer)
  if "face" in FLAGS.model:
    global_step = evaluate_mfacenet.get_global_step()
    evaluate_mfacenet.compute_megaface_metrics(estimator, global_step, writer)
    for dataset in ["lfw", "agedb_30"]:
      evaluate_mfacenet.compute_metrics(dataset, estimator, global_step, writer)


def log_error():
  with open('errors.log', 'a') as fout:
    fout.write(('#' * 20) + '\n')
    fout.write(str(datetime.datetime.now()) + '\n')
    traceback.print_exc(file=fout)

if __name__ == '__main__':
  try:
    absl.app.run(main)
  except tf.errors.ResourceExhaustedError as E:
    print("#" * 20, "\nCaught ResourceExhaustedError. Re-raising exception ...\n", "#" * 20)
    log_error()
    raise E
  except tf.errors.InternalError as E:
    print("#" * 20, "\nCaught InternalError. Re-raising exception ...\n", "#" * 20)
    log_error()
    raise E
  except Exception as E:
    print("*" * 20, "\nCaught exception", type(E), "Exiting now ...")
    traceback.print_exc()
