import pprint
import time
import json
from tqdm import tqdm
import sys
import os
import numpy as np
import shutil
import glob
from sklearn.model_selection import KFold
import pickle
from copy import deepcopy
import multiprocessing
import h5py

from absl import flags
from absl import app

flags.DEFINE_string(name='dense_suffix',
                    help='suffix of the dense embedding files',
                    default='dense_resnet_512_run_1.hdf5')
flags.DEFINE_string(name='sparse_suffix',
                    help='suffix of the sparse embedding files',
                    default=None)
flags.DEFINE_float(name='threshold',
                   help='Threshold used to filter the examples',
                   default=None)
flags.DEFINE_integer(name='rerank_topk',
                   help='topk elements to rerank',
                   default=1000)


FLAGS = flags.FLAGS

def compute_recall(sorted_labels, correct_label, top_k):
    scores = [[] for _ in top_k]
    incorrect_seen = 0
    for label in sorted_labels:
      if label != -1 and label != correct_label:
        continue
      if label != correct_label:
        incorrect_seen += 1
      else:
        for i, k in enumerate(top_k):
          scores[i].append(1 if incorrect_seen < k else 0)
    scores = np.mean(scores, axis=1)
    return scores


def get_top_k_idx(labels, scores, k):
    count = 0
    for i, label in enumerate(labels):
        if label == -1:
            count += 1
        if count >= k:
            return i
        if scores[i] < FLAGS.threshold:
          return i


def get_recall_k(megaface_sparse, megaface_dense, facescrub_sparse, facescrub_dense, facescrub_labels, top_k, rerank_k):
  sparse_database = np.concatenate((facescrub_sparse, megaface_sparse))
  dense_database = np.concatenate((facescrub_dense, megaface_dense))
  dlabels = np.concatenate((facescrub_labels, [-1] * len(megaface_sparse)))
  num_probes = len(facescrub_labels)
  print("Num probes", num_probes)
  print("Multiplying matrices")
  dot = np.matmul(facescrub_sparse, sparse_database.T)
  print("Sorting the matrix")
  ranked_indices = np.argsort(-dot, axis=1)
  recalls = []
  it = tqdm(range(num_probes))
  for i in it:
    ranked = ranked_indices[i]
    labels = dlabels[ranked]
    scores = dot[i][ranked]
    if rerank_k > 0:
      print("Re-ranking top-%d" % rerank_k)
      top_idx = get_top_k_idx(labels, scores, rerank_k)
      labels, rest = labels[:top_idx+1], labels[top_idx+1:]
      ranked = ranked[:top_idx+1]
      dense_emb = dense_database[ranked]
      dense_dot = np.matmul(facescrub_dense[i], dense_emb.T).ravel()
      reranked_indices = np.argsort(-dense_dot)
      reranked_labels = labels[reranked_indices]
      labels = np.concatenate((reranked_labels, rest))
    recalls.append(compute_recall(labels, facescrub_labels[i], top_k))
    it.set_description('Recalls %.3f %.3f %.3f %.3f' % tuple(np.mean(recalls, axis=0)))

  return np.mean(recalls, axis=0)


def main(_):
  print("Official Megaface Metrics")

  megaface_sparse = h5py.File('embeddings_reranking/megaface_' + FLAGS.sparse_suffix, 'r')['embedding']
  megaface_dense = h5py.File('embeddings_reranking/megaface_' + FLAGS.dense_suffix, 'r')['embedding']

  facescrub_sparse = h5py.File('embeddings_reranking/facescrub_' + FLAGS.sparse_suffix, 'r')['embedding']
  facescrub_dense = h5py.File('embeddings_reranking/facescrub_' + FLAGS.dense_suffix, 'r')['embedding']
  facescrub_labels = np.asarray(h5py.File('embeddings_reranking/facescrub_' + FLAGS.sparse_suffix, 'r')['label'])
  facescrub_dense_labels = np.asarray(
        h5py.File('embeddings_reranking/facescrub_' + FLAGS.dense_suffix, 'r')['label'])
  assert(np.all(facescrub_labels == facescrub_dense_labels))

  nnz = np.abs(megaface_sparse) >= 1e-8
  sparsity = np.mean(nnz)
  flops = np.mean(nnz, axis=0)
  flops = np.sum(flops * flops)

  mean_nnz = np.mean(np.sum(nnz, axis=1))
  flops_ratio = flops * nnz.shape[1] / (mean_nnz**2)

  print("\tSparsity", sparsity, "Flops", flops, "Ratio", flops_ratio)

  top_k = [1, 10, 50, 100]
  rerank_k = FLAGS.rerank_topk
  print("Rerank TOP_K", rerank_k)
  recalls = get_recall_k(megaface_sparse, megaface_dense,
                         facescrub_sparse, facescrub_dense, facescrub_labels, top_k, rerank_k)

  print("Non Binarized")
  for recall, k in zip(recalls, top_k):
    print("\trecall@%d" % k, recall)


if __name__ == '__main__':
  app.run(main)
