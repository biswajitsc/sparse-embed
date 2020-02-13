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
import json

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
flags.DEFINE_string(name='filelist',
                    help='The list of evaluation files',
                    default=None)             
flags.DEFINE_string(name='prefix',
                    help='appended prefix for the hdf5 file',
                    default=None)       

flags.DEFINE_boolean(name='debug',
                     help='If True, inference will be stopped at 10000 steps',
                     default=True)
flags.DEFINE_boolean(name='is_training',
                     help='If True, embeddings for the training data will be computed',
                     default=False)


FLAGS = flags.FLAGS


def replace_extension(filename):
  pos = filename.rfind(b'.')
  if pos != -1:
    filename = filename[:pos+1]
  filename = filename + b'png'
  return filename

def main(_):
  with open(FLAGS.filelist, 'r') as fin:
    filelist = json.load(fin)
  filelist = filelist['path']
  filelist = [np.string_(os.path.basename(filename)) for filename in filelist]
  if FLAGS.prefix == 'facescrub':
    filelist = [filename.replace(b" ", b"_") for filename in filelist]
    filelist = [replace_extension(filename) for filename in filelist]

  file_dict = {filename:i for i, filename in enumerate(filelist)}
  num_samples = len(filelist)

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

  if not os.path.exists('embeddings_reranking'):
    os.mkdir('embeddings_reranking')

  if FLAGS.output is None:
    output_file = FLAGS.prefix + "_" + os.path.basename(FLAGS.model_dir.rstrip('/')) + '.hdf5'
  else:
    output_file = FLAGS.output
  output_file = os.path.join('embeddings_reranking', output_file)
  if os.path.exists(output_file):
    print('%s exists. Exiting' % output_file)
    sys.exit()

  print("*" * 10, "Writing to", output_file)

  sparsity = []
  embeddings = []
  labels = []
  idx = -np.ones(num_samples, dtype=np.int32)

  for step, prediction in enumerate(predictions):
    embedding = prediction["small_embeddings"]
    true_label = prediction["true_labels"][0]
    true_label_text = np.string_(prediction["true_label_texts"])
    filename = np.string_(prediction["filename"])

    idxs = np.where(np.abs(embedding) >= FLAGS.zero_threshold)[0]
    sparsity.append(len(idxs))

    embeddings.append(embedding)
    labels.append(true_label)

    id = file_dict[filename]
    assert(idx[id] == -1)
    idx[id] = step

    if step % 10000 == 0:
      print('Step %d Av. Sparsity %f' % (step, np.mean(sparsity)))
      sys.stdout.flush()
    if FLAGS.debug and step >= 10000:
      break
  
  embeddings = np.asarray(embeddings)
  embeddings = embeddings[idx]
  print("embeddings shape", embeddings.shape)

  labels = np.asarray(labels)
  labels = labels[idx]
  print("labels shape", labels.shape)


  h5file = h5py.File(output_file, mode='w')
  h5file.create_dataset(name='embedding', data=embeddings)
  h5file.create_dataset(name='label', data=labels)
  h5file.close()
  print("Done writing to", output_file)


if __name__ == '__main__':
  absl.app.run(main)
