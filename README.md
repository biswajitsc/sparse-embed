# sparse-embed
Code for paper 'Minimizing FLOPs to Learn Efficient Sparse Representations' published at ICLR 2020 https://openreview.net/forum?id=SygpC6Ntvr

The main training and testing code is based on [TensorFlow](https://www.tensorflow.org/), and multi-GPU training is achieved using [Horovod](https://github.com/horovod/horovod).

## Requirements
The code has been tested to work with the following versions:
* python 3.6
* tensorflow-gpu 1.12
* horovod 0.16


## Datasets
**Training data:** The MS1M dataset for training was downloaded from the InsightFace [repository](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo). The dataset is aligned by authors of InsightFace using [MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html).

The downloaded dataset is in MXnet format, and must be converted to JPEG images using ``python preprocessing/msceleb_to_jpeg.py <dataset_directory>``. The extracted images will be stored in ``dataset_directory/images``.

**Testing data:** The MegaFace and FaceScrub datasets can been downloaded from the official MegaFace challenge [website](http://megaface.cs.washington.edu/participate/challenge.html). We used [MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html) to align the images.

The CIFAR-100 training and testing data can be downloaded from https://www.cs.toronto.edu/~kriz/cifar.html.

## Preprocessing
All the datasets must be converted to TFRecords format for fast and multi-threaded pre-fetching of data. The scripts in ``preprocessing/*.sh`` can be used to convert to TFRecords format. The output directories can be modified within the code. The preprocessing code has been taken from the official TF models [github](https://github.com/tensorflow/models).

### Training
The pipeline to read images (``input_pipeline.py``, ``imagenet_preprocessing.py``) during training has also been taken from the TF models [github](https://github.com/tensorflow/models), and the main models have been implemented using the [tf.Estimator](https://www.tensorflow.org/guide/estimator) framework. Sample training scripts can be found in ``run_scripts/face`` and ``run_scripts/cifar``. The directories for the trained models and the dataset directories can be specified withing the shell scripts.

### Evaluation
For evaluation, we have to first compute the trained embeddings for FaceScrub and MegaFace. Example scripts using ``get_embeddings_reranking.py`` are given in ``run_scripts/get_embeddings``. The computed embeddings will be saved as [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) files in a directory named ``embeddings_reranking``. The MegaFace and FaceScrub embeddings start with the prefixes ``megaface`` and ``facescrub`` respectively. For instance, the MegaFace embeddings corresponding to the model ``sparse_mobilenet_fl_run_1`` will be saved as ``megaface_sparse_mobilenet_fl_run_1.hdf5``.

The accuracy and retrieval times are evaluated independently as described below.

**Accuracy:** Accuracy can be evaluated as
```
python evaluate_mfacenet_reranking.py --dense_suffix=dense_mobilenet_512_run_1.hdf5 --sparse_suffix=sparse_mobilenet_fl_run_1.hdf5 --threshold=0.25 --rerank_topk=1000
```
``dense_suffix`` and ``sparse_suffix`` parameters denote the corresponding and dense and sparse embeddings to use. The dense embeddings are used for re-ranking as described in the paper. The ``threshold`` and ``rerank_topk`` parameters denote the filtering threshold and the number of top candidates to rerank, respectively. Refer to the paper for a more detailed description of these parameters.

**Retrieval Time:** For measuring the time, a more efficient C++ script is used. The HDF5 files must be converted to binary files before they can be read by the script. ``mobilefacenet_to_bin.sh`` and ``resnet_to_bin.sh`` provide examples to convert the HDF5 files to binary format and saved in the directory ``embedding_bin``. The retrieval time can be measured using ``search/search.cpp``. The makefile ``search/Makefile`` provides a sample flow to compile the script and measure the time retrieval time in ms. The ``runmegaface`` command measures the time using MegaFace distractors and held-out MegaFace queries. The ``runfacescrub`` command on the other hand uses MegaFace distractors and FaceScrub queries.

