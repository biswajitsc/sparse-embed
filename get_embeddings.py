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
import h5py

# Define all necessary hyperparameters as flags
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

flags.DEFINE_string(name='model_dir',
                    help='Path to the model directory for checkpoints and summaries',
                    default=None)
flags.DEFINE_string(name='data_dir',
                    help='Directory containing the validation data',
                    default=None)
flags.DEFINE_string(name='model',
                    help='Options are mobilenet and mobilefacenet',
                    default=None)
flags.DEFINE_string(name='output',
                    help='The output filename',
                    default=None)
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


def main(_):
  config = tf.estimator.RunConfig(
    model_dir=FLAGS.model_dir,
    keep_checkpoint_every_n_hours=1,
    save_summary_steps=50,
    save_checkpoints_secs=60 * 20,
  )
  estimator = tf.estimator.Estimator(
    model_fn=model.model_fn,
    config=config
  )

  predictions = estimator.predict(
    input_fn=lambda: model.imagenet_iterator(is_training=FLAGS.is_training),
    yield_single_examples=True,
    predict_keys=["true_labels", "true_label_texts", "small_embeddings", "filename"]
  )

  if not os.path.exists('embeddings'):
    os.mkdir('embeddings')

  if FLAGS.output is None:
    output_file = os.path.basename(FLAGS.model_dir.rstrip('/')) + '.hdf5'
  else:
    output_file = FLAGS.output
  output_file = os.path.join('embeddings', output_file)
  if os.path.exists(output_file):
    print('%s exists. Exiting' % output_file)
    sys.exit()

  print("Writing to", output_file)
  h5file = h5py.File(output_file, mode='w')
  inttype = h5py.special_dtype(vlen=np.int32)
  floattype = h5py.special_dtype(vlen=np.float32)

  maxlen = int(1.1e6)
  embedding_vals = h5file.create_dataset(name='embedding_vals',
      shape=(maxlen,), dtype=floattype)
  embedding_idx = h5file.create_dataset(name='embedding_idx',
      shape=(maxlen,), dtype=inttype)
  # embedding_dense = h5file.create_dataset(name='embedding_dense',
  #     shape=(maxlen,), dtype=floattype)
  labels = h5file.create_dataset(name='labels', shape=(maxlen,), dtype=np.int32)
  # text_labels = h5file.create_dataset(name='text_labels', shape=(maxlen,), dtype='S50')
  # filenames = h5file.create_dataset(name='filenames', shape=(maxlen,), dtype='S50')
  tot_embeddings = h5file.create_dataset(name='tot_len', shape=(1,), dtype=np.int32)

  sparsity = []
  start_time = time.time()  
  for step, prediction in enumerate(predictions):
    embedding = prediction["small_embeddings"]
    true_label = prediction["true_labels"][0]
    true_label_text = np.string_(prediction["true_label_texts"])
    filename = np.string_(prediction["filename"])

    print(filename)

    idxs = np.where(np.abs(embedding) >= FLAGS.zero_threshold)[0]
    sparsity.append(len(idxs))

    embedding_vals[step] = [embedding[idx] for idx in idxs]
    embedding_idx[step] = [idx for idx in idxs]
    # embedding_dense[step] = [emb for emb in embedding]
    labels[step] = true_label
    # text_labels[step] = true_label_text
    # filenames[step] = filename
    tot_embeddings[0] = step+1

    if step % 10000 == 0:
      print('Step %d Av. Sparsity %f' % (step, np.mean(sparsity)))
      sys.stdout.flush()
    if FLAGS.debug and step >= 10000:
      break
    # if step >= 300000:
    #   break
  inference_time = time.time() - start_time
  print("Inference time %f mins" % (inference_time / 60.0))

  h5file.close()
  print("Done writing to", output_file)


if __name__ == '__main__':
  absl.app.run(main)
