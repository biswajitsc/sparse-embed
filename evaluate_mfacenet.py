from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import model
import input_pipeline
import pprint
import time
import json
from tqdm import tqdm
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
# import mxnet as mx
import flags
from copy import deepcopy
import multiprocessing

# Define all necessary hyperparameters as flags
FLAGS = absl.flags.FLAGS
FAR = 1e-3


def img_full_path(filename):
  synset = filename.split('_')[0]
  full_path = os.path.join('../imagenet10k/images', synset, filename)
  return full_path


def get_global_step(ckpt_path=None):
  if ckpt_path is None:
    ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
    print(FLAGS.model_dir)
    print(ckpt_path)
  base = os.path.basename(ckpt_path)
  global_step = int(base.split('-')[1])
  return global_step


def get_sorted_global_steps():
  all_paths = glob.glob(os.path.join(FLAGS.model_dir, '*.ckpt*data*'))
  global_steps = [int(path.split('-')[-4].split('.')[0]) for path in all_paths]
  global_steps.sort()
  return global_steps


def get_last_eval_step():
  try:
    last_step = np.loadtxt(os.path.join(FLAGS.model_dir, 'eval', 'last_eval.txt'))
    return last_step
  except OSError:
    save_last_eval_step(0)
    return 0


def save_last_eval_step(last_step):
  file_path = os.path.join(FLAGS.model_dir, 'eval', 'last_eval.txt')
  np.savetxt(file_path, [last_step])
  print('Saved last_step=%d to %s' % (last_step, file_path))


def load_data(records_dir, record_name):
  lfw_bin = os.path.join(records_dir, record_name + ".pkl")
  bins, labels = pickle.load(open(lfw_bin, "rb"), encoding='bytes')
  dist_bins = np.unique(bins, axis=0)
  return dist_bins, bins, labels


def binarize(mat):
  mat = deepcopy(mat)
  mat[mat > FLAGS.zero_threshold] = 1.0
  mat[mat < FLAGS.zero_threshold] = -1.0
  norm = np.linalg.norm(mat, axis=1, keepdims=True)
  mat = mat / norm
  return mat

###################################### LFW and AgeDB metrics ###########################

def eval_input_fn(bins):
  dummy_labels = tf.constant([1] * len(bins))
  bins = tf.constant(bins)
  dataset = tf.data.Dataset.from_tensor_slices((bins, dummy_labels))

  def _parse_function(image, label):
    # image = tf.image.decode_jpeg(img_bin, channels=3)
    # image.set_shape([112, 112, 3])
    image = tf.cast(image, dtype=tf.float32)
    image = image / 128.0 - 1.0
    return image, label

  dataset = dataset.map(_parse_function)
  dataset = dataset.batch(FLAGS.batch_size)

  iterator = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()
  feature_dict = {
    "features": features,
    "labels": labels,
    "label_texts": labels,
    "filename": labels
  }

  return feature_dict, None


def get_accuracy(embeddings, labels):
  emb1 = embeddings[:, :, 0, :].squeeze()
  emb2 = embeddings[:, :, 1, :].squeeze()
  dists = np.sum(emb1 * emb2, axis=-1)  # [10, 600]

  labels = np.asarray(labels).reshape(10, 600)

  accuracies = []

  for split in range(10):
    dists_subset = np.concatenate([dists[:split], dists[split+1:]])
    labels_subset = np.concatenate([labels[:split], labels[split+1:]])

    dists_subset = dists_subset.ravel()
    labels_subset = labels_subset.ravel()

    pairs = list(zip(dists_subset, labels_subset))
    pairs.sort(key=lambda pair: pair[0])

    best_correct = 0
    best_thresh = -1.0
    correct_count = np.sum(labels_subset)
    print("init correct_count %d" % correct_count)
    for pred, true in pairs:
      if true:
        correct_count -= 1
      else:
        correct_count += 1

      if correct_count > best_correct:
        best_correct = correct_count
        best_thresh = pred

    print("Split %d" % split)
    print("\tTrain: Threshold %f\tAccuracy %f" % (best_thresh, best_correct * 1.0 / len(pairs)))

    dists_subset = dists[split]
    labels_subset = labels[split]

    correct_count = np.sum((dists_subset >= best_thresh) == labels_subset)
    accuracy = correct_count * 1.0 / len(dists_subset)
    print("\tTest: Accuracy %f" % accuracy)
    accuracies.append(accuracy)
  
  return accuracies, dists.ravel(), labels.ravel()


def save_distances(dists, labels, dataset):
  dirname = 'face_verification_results'
  os.makedirs(dirname, exist_ok=True)
  basename = os.path.basename(FLAGS.model_dir)
  filename = dataset + '_' + basename + '.npz'
  np.savez(os.path.join(dirname, filename), dists=dists, labels=labels)


def compute_metrics(dataset_name, estimator, global_step, summary_writer=None):
  print(dataset_name.upper())
  dist_bins, bins, labels = load_data(FLAGS.lfw_dir, dataset_name)

  dist_predictions = estimator.predict(
    input_fn=lambda: eval_input_fn(dist_bins),
    yield_single_examples=True,
    predict_keys=["small_embeddings"],
    checkpoint_path=os.path.join(FLAGS.model_dir, 'model.ckpt-%d' % global_step)
  )

  dist_embeddings = []
  for prediction in dist_predictions:
    embedding = prediction["small_embeddings"]
    dist_embeddings.append(embedding)
  print("Computed %d distinct embeddings" % len(dist_embeddings))
  dist_embeddings = np.asarray(dist_embeddings)

  predictions = estimator.predict(
    input_fn=lambda: eval_input_fn(bins),
    yield_single_examples=True,
    predict_keys=["small_embeddings"],
    checkpoint_path=os.path.join(FLAGS.model_dir, 'model.ckpt-%d' % global_step)
  )

  embeddings = []
  for prediction in predictions:
    embedding = prediction["small_embeddings"]
    embeddings.append(embedding)
  print("Computed %d embeddings" % len(embeddings))
  embeddings = np.asarray(embeddings)
  embeddings[np.abs(embeddings) <= FLAGS.zero_threshold] = 0.0
  
  binarized_emb = binarize(embeddings)

  embeddings = embeddings.reshape((10, 600, 2, -1))
  binarized_emb = binarized_emb.reshape((10, 600, 2, -1))

  accuracies, all_dists, all_labels = get_accuracy(embeddings, labels)
  save_distances(all_dists, all_labels, dataset_name)
  binarized_acc, _, _ = get_accuracy(binarized_emb, labels)

  print("\n%s\tGlobal Step %d" % (dataset_name.upper(), global_step))
  mean = np.mean(accuracies)
  std = np.std(accuracies)
  print("Final accuracy")
  print("\tMean %f\tStdev %f" % (mean, std))

  bin_mean = np.mean(binarized_acc)
  bin_std = np.std(binarized_acc)
  print("Final binarized accuracy")
  print("\tMean %f\tStdev %f" % (bin_mean, bin_std))

  row_sparsity = np.mean(np.abs(dist_embeddings) <= FLAGS.zero_threshold, axis=1)
  print("Row sparsity")
  print("\tmean", np.mean(row_sparsity), "std", np.std(row_sparsity))

  col_sparsity = np.mean(np.abs(dist_embeddings) <= FLAGS.zero_threshold, axis=0)
  print("Column sparsity")
  print("\tmean", np.mean(col_sparsity), "std", np.std(col_sparsity))

  sparsity = np.mean(np.sum(np.abs(dist_embeddings) >= FLAGS.zero_threshold, axis=1))
  flops = np.sum(np.abs(dist_embeddings) >= FLAGS.zero_threshold, axis=0)
  num_samples = len(dist_embeddings)
  flops = np.sum(flops * (flops - 1)) / (num_samples * (num_samples - 1))
  print("Flops %f" % flops)

  sys.stdout.flush()
  if summary_writer is not None:
    summary = tf.Summary(value=[
        tf.Summary.Value(tag='accuracy/' + dataset_name, simple_value=mean),
        tf.Summary.Value(tag='accuracy/' + dataset_name + '_bin', simple_value=bin_mean),
        # tf.Summary.Value(tag='sparsity/small', simple_value=sparsity),
        # tf.Summary.Value(tag='sparsity/flops', simple_value=flops),
      ])

    summary_writer.add_summary(summary, global_step=global_step)




################################ Megaface metrics ###################################

def megaface_file_input_fn(dataset='megaface'):
  if dataset == 'megaface':
    with open(FLAGS.megaface_list, 'r') as fin:
      filelist = json.load(fin)["path"]
    filelist = [os.path.join(FLAGS.megaface_dir, f) for f in filelist]
    labels = tf.constant([1] * len(filelist))
  
  elif dataset == 'facescrub':
    with open(FLAGS.facescrub_list, 'r') as fin:
      paths = json.load(fin)
      filelist = paths["path"]
      labels = paths["id"]
    
    label_dict = {}
    label_int = []
    for label in labels:
      if label not in label_dict:
        label_dict[label] = len(label_dict)
      label_int.append(label_dict[label])
    filelist = [os.path.join(FLAGS.facescrub_dir, f) for f in filelist]
    labels = tf.constant(label_int)
  
  filelist = tf.constant(filelist)
  dataset = tf.data.Dataset.from_tensor_slices((filelist, labels))

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


def megaface_tfr_input_fn(dataset="megaface"):
  if dataset == "megaface":
    data_dir = FLAGS.megaface_dir
  elif dataset == "facescrub":
    data_dir = FLAGS.facescrub_dir
  
  input_pipeline._DEFAULT_IMAGE_SIZE = 112
  imagenet_preprocessing._RESIZE_MIN = 112

  data = input_pipeline.input_fn(
    is_training=False,
    data_dir=data_dir,
    batch_size=FLAGS.batch_size,
    num_epochs=1,
    return_label_text=True,
    filter_fn=None,
    fixed_crop=True
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


def compute_recall(ranks, correct_label, dlabels, top_k):
    scores = [[] for _ in top_k]
    incorrect_seen = 0
    for idx in ranks[1:]:
      if dlabels[idx] != -1 and dlabels[idx] != correct_label:
        continue
      if dlabels[idx] != correct_label:
        incorrect_seen += 1
      else:
        for i, k in enumerate(top_k):
          scores[i].append(1 if incorrect_seen < k else 0)
    scores = np.mean(scores, axis=1)
    return scores


def get_recall_k(distractors, probes, probe_labels, top_k):
  database = np.concatenate((distractors, probes))
  dlabels = np.concatenate(([-1] * len(distractors), probe_labels))
  num_probes = len(probes)
  print("Multiplying matrices")
  dot = np.matmul(probes, database.T)
  print("Sorting the matrix")
  ranked = np.argsort(-dot, axis=1)
  print("Computing simple scores in parallel")
  if 'autonlab' not in os.uname()[1]:
    pool = multiprocessing.Pool(16)
    recalls = pool.starmap(compute_recall,
                   zip(ranked, probe_labels, [dlabels] * num_probes, [top_k] * num_probes))
  else:
    print("Not using MULTIPROCESSING")
    recalls = [compute_recall(ranks, label, dlabels, top_k) \
                for ranks, label in zip(ranked, probe_labels)]
  return np.mean(recalls, axis=0)


def compute_megaface_metrics(estimator, global_step, summary_writer=None, num_distractors=10000):
  print("Official Megaface Metrics")
  print("Computing for global step %d" % global_step)
  predictions = estimator.predict(
    input_fn=lambda: megaface_tfr_input_fn("megaface"),
    yield_single_examples=True,
    predict_keys=["true_labels", "true_label_texts", "small_embeddings", "filename"],
    checkpoint_path=os.path.join(FLAGS.model_dir, 'model.ckpt-%d' % global_step)
  )

  megaface_embeddings = []

  step = 0
  for prediction in tqdm(predictions):
    embedding = prediction["small_embeddings"]
    megaface_embeddings.append(embedding)

    step += 1
    if step >= num_distractors:
      break
    if FLAGS.debug and step >= 10:
      break

  megaface_embeddings = np.vstack(megaface_embeddings)
  megaface_embeddings[np.abs(megaface_embeddings) <= FLAGS.zero_threshold] = 0.0
  print("\nDistractors Shape", megaface_embeddings.shape)

  predictions = estimator.predict(
    input_fn=lambda: megaface_tfr_input_fn("facescrub"),
    yield_single_examples=True,
    predict_keys=["true_labels", "true_label_texts", "small_embeddings", "filename"],
    checkpoint_path=os.path.join(FLAGS.model_dir, 'model.ckpt-%d' % global_step)
  )

  facescrub_embeddings = []
  labels = []

  for prediction in tqdm(predictions):
    true_label = prediction["true_labels"].ravel()[0]
    embedding = prediction["small_embeddings"]
    facescrub_embeddings.append(embedding)
    labels.append(true_label)

  facescrub_embeddings = np.vstack(facescrub_embeddings)
  facescrub_embeddings[np.abs(facescrub_embeddings) <= FLAGS.zero_threshold] = 0.0
  print("\nProbes Shape", facescrub_embeddings.shape)

  nnz = np.abs(megaface_embeddings) >= FLAGS.zero_threshold
  sparsity = 1.0 - np.mean(nnz)
  flops = np.mean(nnz, axis=0)
  flops = np.sum(flops * flops)

  mean_nnz = np.mean(np.sum(nnz, axis=1))
  flops_ratio = flops * FLAGS.embedding_size / (mean_nnz**2)

  print("\tSparsity", sparsity, "Flops", flops, "Ratio", flops_ratio)

  bin_megaface = binarize(megaface_embeddings)
  bin_facescrub = binarize(facescrub_embeddings)

  top_k = [1, 10, 50, 100]
  recalls = get_recall_k(megaface_embeddings, facescrub_embeddings, labels, top_k)

  print("Non Binarized")
  for recall, k in zip(recalls, top_k):
    print("\trecall@%d" % k, recall)

  bin_recalls = get_recall_k(bin_megaface, bin_facescrub, labels, top_k)

  print("Binarized")
  for recall, k in zip(bin_recalls, top_k):
    print("\trecall@%d" % k, recall)

  if summary_writer is not None:
    summary = tf.Summary(value=
        [tf.Summary.Value(tag='accuracy/recall_%d' % k, simple_value=recall) for recall, k in zip(recalls, top_k)] + \
        [tf.Summary.Value(tag='accuracy/bin_recall_%d' % k, simple_value=recall) for recall, k in zip(bin_recalls, top_k)] + \
        [tf.Summary.Value(tag='sparsity/small', simple_value=mean_nnz),
         tf.Summary.Value(tag='sparsity/flops', simple_value=flops),
         tf.Summary.Value(tag='sparsity/ratio', simple_value=flops_ratio)]
    )
    summary_writer.add_summary(summary, global_step=global_step)


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

  global_step = get_global_step()
  compute_metrics("lfw", estimator, global_step)
  compute_metrics("agedb_30", estimator, global_step)
  print("*** Megaface evaluation is disabled ***")
  # compute_megaface_metrics(estimator, global_step, num_distractors=1e10)

if __name__ == '__main__':
  absl.app.run(main)
