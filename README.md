# sparse-embed
Code for paper 'Minimizing FLOPs to Learn Efficient Sparse Representations' published at ICLR 2020 https://openreview.net/forum?id=SygpC6Ntvr

The main training and testing code is based on [TensorFlow](https://www.tensorflow.org/), and multi-GPU training is achieved using [Horovod](https://github.com/horovod/horovod). Any questions or comments can be mailed to the first author.

### Requirements
The code has been tested to work with the following versions:
* python 3.6
* tensorflow-gpu 1.12
* horovod 0.16


### Datasets
**Training data:** The MS1M dataset for training was downloaded from the InsightFace [repository](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo). The dataset is aligned by InsightFace authors using [MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html).

The downloaded dataset is in MXnet format. Use ``python preprocessing/msceleb_to_jpeg.py dataset_directory`` to extract jpeg images. Extracted images will be stored in ``dataset_directory/images``.

**Testing data:** The MegaFace and FaceScrub datasets can been downloaded from the official MegaFace challenge [website](http://megaface.cs.washington.edu/participate/challenge.html). We used [MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html) to align the images.

The CIFAR-100 training and testing data can be downloaded from https://www.cs.toronto.edu/~kriz/cifar.html.

### Preprocessing
All the datasets must be converted to TFRecords format for fast and multi-threaded pre-fetching of data. Use ``preprocessing/*.sh`` to convert to TFRecords format. The output directories can be easily modified within the code. The preprocessing code has been taken from the official TF models [github](https://github.com/tensorflow/models).

### Training
The pipeline to read images (``input_pipeline.py``, ``imagenet_preprocessing.py``) during training has also been taken from the TF models [github](https://github.com/tensorflow/models). Our models have been implemented using the [tf.Estimator](https://www.tensorflow.org/guide/estimator) framework. Sample training scripts have been provided in ``run_scripts/face`` and ``run_scripts/cifar``. The directories for the trained models and the dataset directories can be easily modified in the code.

### Evaluation
For evaluation, the trained embeddings for FaceScrub and MegaFace must be extracted first. Example scripts using ``get_embeddings_reranking.py`` are given in  ``run_scripts/get_embeddings``. The dense and sparse embeddings will be saved as [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) files in a directory named ``embeddings_reranking``. The MegaFace and FaceScrub embeddings start with the prefixes ``megaface`` and ``facescrub`` respectively. For instance, the MegaFace embeddings corresponding to the model ``sparse_mobilenet_fl_run_1`` will be saved as ``megaface_sparse_mobilenet_fl_run_1.hdf5``. Accuracy and retrieval time are evaluated separately using these embeddings.

**Accuracy:** Accuracy can be evaluated as
```
python evaluate_mfacenet_reranking.py --dense_suffix=dense_mobilenet_512_run_1.hdf5 --sparse_suffix=sparse_mobilenet_fl_run_1.hdf5 --threshold=0.25 --rerank_topk=1000
```
``dense_suffix`` and ``sparse_suffix`` parameters denote the corresponding and dense and sparse embeddings to use. The dense embeddings are used for re-ranking and described in the paper. The ``threshold`` and ``rerank_topk`` parameters denote the filtering threshold and the number of top candidates to rerank, respectively. Refer to the paper for a more detailed description of these parameters.

**Retrieval Time:** For measuring the time, a more efficient C++ script is used. The HDF5 files must be converted to binary files before they can be read by the script. ``mobilefacenet_to_bin.sh`` and ``resnet_to_bin.sh`` provide examples to convert the HDF5 files to binary format and saved in the directory ``embedding_bin``. The retrieval time can be measured using ``search/search.cpp``. The makefile ``search/Makefile`` provides a sample flow to compile the script and measure the time retrieval time in ms. The ``runmegaface`` command measures the time using MegaFace distractors and held-out MegaFace queries. The ``runfacescrub`` command on the other hand uses MegaFace distractors and FaceScrub queries.

