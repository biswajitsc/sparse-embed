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
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
import sklearn
import traceback
import logging
import scipy
import flags
from tqdm import tqdm
from copy import deepcopy
from cifar import cifar

# Define all necessary hyperparameters as flags
FLAGS = absl.flags.FLAGS

def save_image(image, filename):
  feature_ = (image + 1.0) * 128
  img = Image.fromarray(np.uint8(feature_), 'RGB')
  img.save(filename)


def img_full_path(filename):
  synset = filename.split('_')[0]
  full_path = os.path.join('../imagenet10k/images', synset, filename)
  return full_path


def get_global_step(ckpt_path=None):
  if ckpt_path is None:
    ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
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


def binarize(mat):
  mat = deepcopy(mat)
  mat[mat > FLAGS.zero_threshold] = 1.0
  mat[mat < FLAGS.zero_threshold] = -1.0
  norm = np.linalg.norm(mat, axis=1, keepdims=True)
  mat = mat / norm
  return mat


def assign_by_euclidian_at_k(X, T, k):
    """
    X : [nb_samples x nb_features], e.g. 100 x 64 (embeddings)
    k : for each sample, assign target labels of k nearest points
    """
    # distances = sklearn.metrics.pairwise.pairwise_distances(X)
    distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X))
    # get nearest points
    indices = np.argsort(distances, axis=1)[:, 1:k + 1]
    return np.array([[T[i] for i in ii] for ii in indices])


def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """
    s = sum([1 for t, y in zip(T, Y) if t in y[:k]])
    return s / (1. * len(T))


def evaluate(X, T, nb_classes):
    # calculate NMI with kmeans clustering
    # nmi = sklearn.metrics.cluster.normalized_mutual_info_score(
    #     sklearn.cluster.KMeans(nb_classes).fit(X).labels_,
    #     T
    # )
    # logging.info("NMI: {:.3f}".format(nmi * 100))

    # get predictions by assigning nearest 8 neighbors with euclidian
    Y = assign_by_euclidian_at_k(X, T, 8)

    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in [1, 2, 4, 8]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        logging.info("R@{} : {:.3f}".format(k, 100 * r_at_k))

    # return nmi, recall


def compute_nmi(embeddings, labels):
  print("Clustering ...")
  num_classes = len(set(labels))
  print("num_classes %d\tnum_examples %d" % (num_classes, len(embeddings)))

  centers = {}
  for emb, label in zip(embeddings, labels):
    if label not in centers:
      centers[label] = (emb, 1)
    else:
      pair = centers[label]
      centers[label] = (pair[0] + emb, pair[1] + 1)
  centers = [centers[key] for key in centers]
  centers = [pair[0] / pair[1] for pair in centers]
  centers = np.asarray(centers)
  print("Computed centers")

  kmeans = MiniBatchKMeans(n_clusters=num_classes, init=centers,
                           batch_size=1000, max_iter=1000)
  kmeans = kmeans.fit(embeddings)
  cluster_labels = kmeans.labels_
  nmi = normalized_mutual_info_score(labels, cluster_labels)
  return nmi

MAX_QUERY = 1000
MAX_DISTRACTORS = 500000
def eval_recall(embedding, label, ks):
    embedding = np.copy(embedding)
    # Normalized embeddings
    embedding /= np.linalg.norm(embedding, axis=1).reshape((-1, 1))
    norm = np.sum(embedding * embedding, axis = 1)
    num_correct = np.zeros(len(ks))
    num_queries = min(embedding.shape[0], MAX_QUERY)
    for i in tqdm(range(num_queries)):
        dis = norm[i] + norm - 2 * np.squeeze(np.matmul(embedding[i],embedding.T))
        dis[i] = 1e10
        pred = np.argsort(dis)
        for j, k in enumerate(ks):
          if label[i] in label[pred[:k]]:
            num_correct[j] += 1
    recall = num_correct / num_queries
    return recall


def eval_precision(embedding, label, ks):
    embedding = np.copy(embedding)
    # Normalized embeddings
    embedding /= np.linalg.norm(embedding, axis=1).reshape((-1, 1))
    num_correct = np.zeros(len(ks))
    num_queries = min(embedding.shape[0], MAX_QUERY)
    for i in tqdm(range(num_queries)):
        dis = - np.squeeze(np.matmul(embedding[i],embedding.T))
        dis[i] = 1e10
        pred = np.argsort(dis)
        for j, k in enumerate(ks):
          num_correct[j] += np.mean(label[pred[:k]] == label[i])
    recall = num_correct / num_queries
    return recall


def compute_metrics(estimator, global_step, is_training, summary_writer=None):
  global MAX_DISTRACTORS
  global MAX_QUERY

  print("Computing for global step %d" % global_step)
  if FLAGS.model == 'cifar100':
    input_fn = lambda: cifar.input_fn(
      FLAGS.data_dir, is_training, FLAGS.batch_size, 1)
  else:
    if not is_training:
      MAX_DISTRACTORS = 1e10
      MAX_QUERY = 10000
    input_fn = lambda: model.imagenet_iterator(is_training=is_training)

  predictions = estimator.predict(
    input_fn=input_fn,
    yield_single_examples=True,
    predict_keys=["true_labels", "true_label_texts", "small_embeddings", "filename"],
    checkpoint_path=os.path.join(FLAGS.model_dir, 'model.ckpt-%d' % global_step)
  )

  true_labels = []
  true_label_texts = []
  small_embeddings = []
  filenames = []

  start_time = time.time()
  step = 0
  for prediction in tqdm(predictions):
    filenames.append(prediction["filename"])
    true_labels.append(prediction["true_labels"][0])
    true_label_texts.append(prediction["true_label_texts"])
    small_embeddings.append(prediction["small_embeddings"])
    step += 1
    if FLAGS.debug and step >= 100000:
      break
    if step >= MAX_DISTRACTORS:
      break
  inference_time = time.time() - start_time

  true_labels = np.asarray(true_labels)
  small_embeddings = np.asarray(small_embeddings)
  small_embeddings[np.abs(small_embeddings) <= FLAGS.zero_threshold] = 0.0
  binarized_embeddings = binarize(small_embeddings)

  ks = [1, 4, 16, 64]
  precisions = eval_precision(small_embeddings, true_labels, ks)
  bin_precisions = eval_precision(binarized_embeddings, true_labels, ks)

  print("Global Step %d" % global_step)
  # print("Max dot product")
  # print("\tmean", np.mean(max_dots), "std", np.std(max_dots))

  metrics_time = time.time() - start_time

  print("Time")
  print("\tInference", inference_time / 60, "minutes")
  print("\tMetrics", metrics_time / 60, "minutes")

  sys.stdout.flush()

  nnz = np.abs(small_embeddings) >= FLAGS.zero_threshold
  sparsity = 1.0 - np.mean(nnz)
  flops = np.mean(nnz, axis=0)
  flops = np.sum(flops * flops)

  mean_nnz = np.mean(np.sum(nnz, axis=1))
  flops_ratio = flops * FLAGS.embedding_size / (mean_nnz**2)
  print("\tSparsity", sparsity, "Flops", flops, "Ratio", flops_ratio)

  print("Precision")
  print("".join(["\t@%d %f" % (ks[i], precisions[i]) for i in range(len(ks))]))

  print("Bin Precision")
  print("".join(["\t@%d %f" % (ks[i], bin_precisions[i]) for i in range(len(ks))]))

  row_sparsity = np.mean(small_embeddings <= FLAGS.zero_threshold, axis=1)
  print("Row sparsity")
  print("\tmean", np.mean(row_sparsity), "std", np.std(row_sparsity))

  col_sparsity = np.mean(small_embeddings <= FLAGS.zero_threshold, axis=0)
  print("Column sparsity")
  print("\tmean", np.mean(col_sparsity), "std", np.std(col_sparsity))

  # print("NMI %f" % nmi)

  if summary_writer is not None:
    summary = tf.Summary(
      value=[tf.Summary.Value(tag='sparsity/small', simple_value=mean_nnz),
             tf.Summary.Value(tag='sparsity/flops', simple_value=flops),
             tf.Summary.Value(tag='sparsity/ratio', simple_value=flops_ratio)] +
      [tf.Summary.Value(tag='accuracy/precision_%d' % (ks[i]), simple_value=precisions[i]) \
          for i in range(len(ks))] +
      [tf.Summary.Value(tag='accuracy/precision_%d_bin' % (ks[i]), simple_value=bin_precisions[i]) \
          for i in range(len(ks))]
    )

    summary_writer.add_summary(summary, global_step=global_step)


def main(_):
  global MAX_DISTRACTORS
  global MAX_QUERY

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

  if not FLAGS.evaluate_all:
    global_step = get_global_step()
    MAX_DISTRACTORS = 1e10
    MAX_QUERY = 10000
    compute_metrics(estimator, global_step, False, None)

  else:
    logdir = os.path.join(FLAGS.model_dir, 'eval')
    print('Writing to', logdir)
    last_eval_time = time.time()
    with tf.summary.FileWriter(
        logdir=logdir,
        flush_secs=30) as writer:
      while True:
        global_steps = get_sorted_global_steps()
        last_eval_step = get_last_eval_step()
        global_steps = [step for step in global_steps if step > last_eval_step]
        if len(global_steps) == 0:
          curr_time = time.time()
          if curr_time - last_eval_time >= 10 * 60:
            print("Nothing else to evaluate. Exiting")
            break
          else:
            print("Waiting for 2 minutes")
            time.sleep(2 * 60)
            continue

        evaluate_next = global_steps[0]
        print("Models to be evaluated", global_steps)
        print("Attempting to evaluate now, Step %d" % evaluate_next)
        try:
          compute_metrics(estimator, evaluate_next, writer)
          save_last_eval_step(evaluate_next)
          last_eval_time = time.time()
        except Exception as e:
          print("*" * 5, "Failed to evaluate Step %d" % evaluate_next)
          traceback.print_exc()


if __name__ == '__main__':
  absl.app.run(main)
