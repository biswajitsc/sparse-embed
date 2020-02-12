"""
The following code has been adapted from
https://github.com/tensorflow/models/blob/master/official/resnet/imagenet_main.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

import imagenet_preprocessing

_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
# _NUM_CLASSES = 1001

# _NUM_IMAGES = {
#     'train': 1281167,
#     'validation': 50000,
# }

_NUM_TRAIN_FILES = 1024
_SHUFFLE_BUFFER_SIZE = 1000

# DATASET_NAME = 'ImageNet'

###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
  """Return filenames for dataset."""
  if is_training:
    # return [
    #     os.path.join(data_dir, 'train-%05d-of-01024' % i)
    #     for i in range(_NUM_TRAIN_FILES)]
    return glob.glob(os.path.join(data_dir, 'train-*'))
  else:
    # return [
    #     os.path.join(data_dir, 'validation-%05d-of-00128' % i)
    #     for i in range(128)]
    return glob.glob(os.path.join(data_dir, 'validation-*'))


def _parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.

  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields (values are included as examples):

    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/object/bbox/xmin: 0.1
    image/object/bbox/xmax: 0.9
    image/object/bbox/ymin: 0.2
    image/object/bbox/ymax: 0.6
    image/object/bbox/label: 615
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.

  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
  """
  # Dense features in Example proto.
  feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                              default_value=-1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
      'image/filename': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
  }
  sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)
  label_text = features['image/class/text']
  filename = features['image/filename']

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  return features['image/encoded'], label, bbox, label_text, filename


def parse_record(raw_record, is_training, return_label_text=True, fixed_crop=False):
  """Parses a record containing a training example of an image.

  The input record is parsed into a label and image, and the image is passed
  through preprocessing steps (cropping, flipping, and so on).

  Args:
    raw_record: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
    is_training: A boolean denoting whether the input is for training.

  Returns:
    Tuple with processed image tensor and one-hot-encoded label tensor.
  """
  image_buffer, label, bbox, label_text, filename = _parse_example_proto(raw_record)

  image = imagenet_preprocessing.preprocess_image(
      image_buffer=image_buffer,
      bbox=bbox,
      output_height=_DEFAULT_IMAGE_SIZE,
      output_width=_DEFAULT_IMAGE_SIZE,
      num_channels=_NUM_CHANNELS,
      is_training=is_training,
      fixed_crop=fixed_crop)

  if return_label_text:
    return image, label, label_text, filename
  else:
    return image, label


def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer,
                           parse_record_fn, num_epochs=1, return_label_text=True,
                           filter_fn=None, fixed_crop=False):
  """Given a Dataset with raw records, return an iterator over the records.

  Args:
    dataset: A Dataset representing raw records
    is_training: A boolean denoting whether the input is for training.
    batch_size: The number of samples per batch.
    shuffle_buffer: The buffer size to use when shuffling records. A larger
      value results in better randomness, but smaller values reduce startup
      time and use less memory.
    parse_record_fn: A function that takes a raw record and returns the
      corresponding (image, label) pair.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    Dataset of (image, label) pairs ready for iteration.
  """

  # We prefetch a batch at a time, This can help smooth out the time taken to
  # load input files as we go through shuffling and processing.
  dataset = dataset.prefetch(buffer_size=batch_size)
  dataset = dataset.map(
    lambda value: parse_record_fn(value, is_training, return_label_text, fixed_crop))
  if filter_fn is not None:
    dataset = dataset.filter(filter_fn)

  # Shuffle the records. Note that we shuffle before repeating to ensure
  # that the shuffling respects epoch boundaries.
  dataset = dataset.shuffle(buffer_size=shuffle_buffer)

  # If we are training over multiple epochs before evaluating, repeat the
  # dataset for the appropriate number of epochs.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
  # allows DistributionStrategies to adjust how many batches to fetch based
  # on how many devices are present.
  dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

  return dataset


def input_fn(is_training, data_dir, batch_size, num_epochs=1,
             return_label_text=True, filter_fn=None, fixed_crop=False):
  """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A Dataset object that can be used for iteration over the input data.
  """
  filenames = get_filenames(is_training, data_dir)
  if len(filenames) == 0:
    raise ValueError("No files found in %s" % data_dir)

  dataset = tf.data.Dataset.from_tensor_slices(filenames)

  # if is_training:
  # Shuffle the input files
  dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

  # Convert to individual records.
  # cycle_length = 10 means 10 files will be read and deserialized in parallel.
  # This number is low enough to not cause too much contention on small systems
  # but high enough to provide the benefits of parallelization. You may want
  # to increase this number if you have a large number of CPU cores.
  dataset = dataset.apply(tf.contrib.data.parallel_interleave(
      tf.data.TFRecordDataset, cycle_length=30))

  return process_record_dataset(
      dataset, is_training, batch_size, _SHUFFLE_BUFFER_SIZE, parse_record,
      num_epochs, return_label_text, filter_fn, fixed_crop
  )
