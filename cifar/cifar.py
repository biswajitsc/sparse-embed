import numpy as np
import tensorflow as tf
import pickle as pkl
import os

import absl
from tqdm import tqdm

flags = absl.flags
FLAGS = flags.FLAGS
slim = tf.contrib.slim


def flip_rl(inputs):
    '''
    Args:
        inputs - 4D tensor [batch, height, width, nchannel]
    Return:
        flips - 4D tensor [batch, height, width, nchannel]

    flips[i][j][width-k][l] = inputs[i][j][k][l]
    '''
    flips = tf.reverse(inputs, axis=[2])
    return flips


def random_flip_rl(inputs):
    '''
    Args: 
        inputs - 4D tensor
    Return: 
        flip left right or original image
    '''
    rand = tf.random_uniform([], minval=-1, maxval=1)
    return tf.cond(tf.greater(rand, 0.0), lambda: inputs, lambda: flip_rl(inputs))


def get_shape(x):
    '''get the shape of tensor as list'''
    return x.get_shape().as_list()


def random_crop(inputs, ch, cw):
    '''
    apply tf.random_crop on 4D
    Args:
        inputs- 4D tensor
        ch - int
            crop height
        cw - int
            crop width
    '''
    ib, ih, iw, ic = get_shape(inputs)
    return tf.random_crop(inputs, [FLAGS.batch_size, ch, cw, ic])


def read_cifar_split(data_dir, train, seen=False):
    if train:
        train_images = np.load(data_dir + '/train_image.npy')
        train_labels = np.load(data_dir + '/train_label.npy')
        
        print('CIFAR-100: images {} labels {}'.format(train_images.shape, train_labels.shape))
        return train_images, train_labels
    
    elif not seen:
        # val_images = np.load(data_dir + '/val_image.npy')
        # val_labels = np.load(data_dir + '/val_label.npy')

        # test_images_u = np.load(data_dir + '/test_image_u.npy')
        # test_labels_u = np.load(data_dir + '/test_label_u.npy')

        # images = np.concatenate((val_images, test_images_u))
        # labels = np.concatenate((val_labels, test_labels_u))

        images = np.load(data_dir + '/test_image_u.npy')
        labels = np.load(data_dir + '/test_label_u.npy')

        print('CIFAR-100: images {} labels {}'.format(images.shape, labels.shape))
        return images, labels

    else:
        images = np.load(data_dir + '/test_image_s.npy')
        labels = np.load(data_dir + '/test_label_s.npy')

        print('CIFAR-100: images {} labels {}'.format(images.shape, labels.shape))

        return images, labels


def input_fn_split(data_dir, train, batch_size, num_epochs, seen=False):
    images, labels = read_cifar_split(data_dir, train, seen)
    labels = np.asarray(labels.reshape((-1, 1)), dtype=np.int32)
    images = np.asarray(images, dtype=np.float32)
    idx = np.random.permutation(len(images))
    images = images[idx]
    labels = labels[idx]
    # images = images / 128.0 # images are already centered

    texts = np.asarray(['-'] * len(labels))

    fn = tf.estimator.inputs.numpy_input_fn(
        {
            "features": images,
            "labels": labels,
            "label_texts": texts,
            "filename": texts
        },
        labels, batch_size=batch_size, num_epochs=num_epochs,
        shuffle=True)
    return fn()

    # texts = tf.constant(['-'] * len(labels))
    # labels = tf.constant(labels, tf.int32)
    # images = tf.constant(images, tf.float32) # channels last
    # dataset = tf.data.Dataset.from_tensor_slices((images, labels, texts, texts))
    # dataset = dataset.repeat(num_epochs)
    # dataset = dataset.batch(batch_size)

    # return dataset


def input_fn(data_dir, train, batch_size, num_epochs):
    with open(os.path.join(data_dir, 'train'), 'rb') as f:
        data = pkl.load(f, encoding='bytes')
        train_images = data[b'data'].astype(np.float32)
        train_labels = data[b'fine_labels']
        image_mean = np.mean(train_images, axis=0)

    with open(os.path.join(data_dir, 'test'), 'rb') as f:
        data = pkl.load(f, encoding='bytes')
        test_images = data[b'data'].astype(np.float32)
        test_labels = data[b'fine_labels']
    
    train_images -= image_mean
    test_images -= image_mean

    train_images /= 128.0
    test_images /= 128.0

    if train:
        images = train_images
        labels = train_labels
    else:
        images = test_images
        labels = test_labels

    labels = np.asarray(labels, dtype=np.int32).reshape((-1, 1))
    images = np.asarray(images, dtype=np.float32)
    images = np.transpose(np.reshape(images, [-1,3,32,32]), [0,2,3,1])
    idx = np.random.permutation(len(images))
    images = images[idx]
    labels = labels[idx]

    texts = np.asarray(['-'] * len(labels))

    fn = tf.estimator.inputs.numpy_input_fn(
        {
            "features": images,
            "labels": labels,
            "label_texts": texts,
            "filename": texts
        },
        labels, batch_size=batch_size, num_epochs=num_epochs,
        shuffle=True)
    return fn()


def proxy_layer(embedding, labels, num_classes):
  with tf.variable_scope('Logits'):
    weights = tf.get_variable(name='proxy_wts',
                              shape=(embedding.shape[-1], num_classes),
                              initializer=tf.random_normal_initializer(stddev=0.001),
                              dtype=tf.float32)
    weights = tf.nn.l2_normalize(weights, axis=0)
    bias = tf.Variable(tf.zeros([num_classes]))
    # alpha = 64.0
    logits = tf.matmul(embedding, weights) # + bias
    # one_hot_labels = tf.one_hot(labels, num_classes)
    # logits -= one_hot_labels * 0.1
  return logits


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


def nin(features, labels, is_training, num_classes):
    if is_training:
        features = random_flip_rl(
                    random_crop(
                        tf.pad(
                            features, [[0,0],[4,4],[4,4],[0,0]], 
                            "CONSTANT"), 32, 32))

    with tf.variable_scope("NIN"):
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=tf.contrib.slim.variance_scaling_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0001)):
            with slim.arg_scope([slim.conv2d], stride=1, padding='SAME', activation_fn=tf.nn.relu):
                with slim.arg_scope(([slim.dropout]), is_training=is_training, keep_prob=0.5):
                    n = features
                    n = slim.conv2d(n, 192, [5, 5], scope='conv2d_0')
                    n = slim.conv2d(n, 160, [1, 1], scope='conv2d_1')
                    n = slim.conv2d(n, 96, [1, 1], scope='conv2d_2')
                    n = slim.max_pool2d(n, [3,3], stride=2, padding='SAME')
                    n = slim.dropout(n)
                    n = slim.conv2d(n, 192, [5, 5], scope='conv2d_3')
                    n = slim.conv2d(n, 192, [1, 1], scope='conv2d_4')
                    n = slim.conv2d(n, 192, [1, 1], scope='conv2d_5')
                    n = slim.avg_pool2d(n, [3,3], stride=2, padding='SAME')
                    n = slim.dropout(n)
                    n = slim.conv2d(n, 192, [3, 3], scope='conv2d_6')
                    n = slim.conv2d(n, 192, [1, 1], activation_fn=None, scope='conv2d_7')

                    n = tf.reduce_mean(n, [1, 2], keepdims=False)
                    n = tf.nn.relu(n)
    
    if FLAGS.final_activation == 'none':
        final_activation = None
    elif FLAGS.final_activation == 'relu':
        final_activation = tf.nn.relu
    elif FLAGS.final_activation == 'soft_thresh':
        final_activation = soft_thresh(0.1)
    else:
        raise ValueError('Unknown final_activation %s' % final_activation)

    with tf.variable_scope('embedding'):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=final_activation,
                            weights_regularizer=slim.l2_regularizer(0.001),
                            biases_initializer=tf.zeros_initializer()):
            n = slim.fully_connected(n, FLAGS.embedding_size, scope = "fc1")
            embedding = tf.nn.l2_normalize(n, axis=1)

    end_points = {}
    end_points['embedding'] = embedding

    logits = proxy_layer(embedding, labels, num_classes)
    end_points['Logits'] = logits

    return logits, end_points


def get_embeddings(estimator, global_step, is_train):
  fn = lambda: input_fn(
    FLAGS.data_dir, is_train, FLAGS.batch_size, 1)

  predictions = estimator.predict(
    input_fn=fn,
    yield_single_examples=True,
    predict_keys=["true_labels", "true_label_texts", "small_embeddings", "filename"],
    checkpoint_path=os.path.join(FLAGS.model_dir, 'model.ckpt-%d' % global_step)
  )

  labels = []
  embeddings = []

  for prediction in tqdm(predictions):
    labels.append(prediction["true_labels"][0])
    embeddings.append(prediction["small_embeddings"])

  labels = np.asarray(labels)
  embeddings = np.asarray(embeddings)
  embeddings[np.abs(embeddings) <= FLAGS.zero_threshold] = 0.0
  return embeddings, labels

def eval_precision(qemb, qlabel, demb, dlabel, ks):
  MAX_QUERY = 10000

  num_correct = np.zeros(len(ks))
  num_queries = min(qemb.shape[0], MAX_QUERY)
  for i in tqdm(range(num_queries)):
      dis = - np.squeeze(np.matmul(qemb[i], demb.T))
      dis[i] = 1e10
      pred = np.argsort(dis)
      for j, k in enumerate(ks):
        num_correct[j] += np.mean(dlabel[pred[:k]] == qlabel[i])
  recall = num_correct / num_queries
  return recall

def compute_metrics(estimator, global_step, is_training, summary_writer=None):
  print("Computing for global step %d" % global_step)

  ks = [1, 4, 16, 64]

  test_embeddings, test_labels = get_embeddings(estimator, global_step, False)
  if is_training:
    train_embeddings, train_labels = get_embeddings(estimator, global_step, True)

  if is_training:
    precisions = eval_precision(test_embeddings, test_labels, train_embeddings, train_labels, ks)
  else:
    precisions = eval_precision(test_embeddings, test_labels, test_embeddings, test_labels, ks)

  print("Global Step %d" % global_step)
  # print("Max dot product")
  # print("\tmean", np.mean(max_dots), "std", np.std(max_dots))

  if is_training:
    embeddings = train_embeddings
  else:
    embeddings = test_embeddings

  nnz = np.abs(embeddings) >= FLAGS.zero_threshold
  sparsity = 1.0 - np.mean(nnz)
  flops = np.mean(nnz, axis=0)
  flops = np.sum(flops * flops)

  mean_nnz = np.mean(np.sum(nnz, axis=1))
  flops_ratio = flops * FLAGS.embedding_size / (mean_nnz**2)
  print("\tSparsity", sparsity, "Flops", flops, "Ratio", flops_ratio)

  print("Precision")
  print("".join(["\t@%d %f" % (ks[i], precisions[i]) for i in range(len(ks))]))

  row_sparsity = np.mean(embeddings <= FLAGS.zero_threshold, axis=1)
  print("Row sparsity")
  print("\tmean", np.mean(row_sparsity), "std", np.std(row_sparsity))

  col_sparsity = np.mean(embeddings <= FLAGS.zero_threshold, axis=0)
  print("Column sparsity")
  print("\tmean", np.mean(col_sparsity), "std", np.std(col_sparsity))

  # print("NMI %f" % nmi)

  if summary_writer is not None:
    summary = tf.Summary(
      value=[tf.Summary.Value(tag='sparsity/small', simple_value=mean_nnz),
             tf.Summary.Value(tag='sparsity/flops', simple_value=flops),
             tf.Summary.Value(tag='sparsity/ratio', simple_value=flops_ratio)] +
      [tf.Summary.Value(tag='accuracy/precision_%d' % (ks[i]), simple_value=precisions[i]) \
          for i in range(len(ks))]
    )

    summary_writer.add_summary(summary, global_step=global_step)
