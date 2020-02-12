# sparse-embed
Code for paper 'Minimizing FLOPs to Learn Efficient Sparse Representations' published at ICLR 2020 https://openreview.net/forum?id=SygpC6Ntvr

The main training and testing code is based on [TensorFlow](https://www.tensorflow.org/), and multi-GPU training is achieved using [Horovod](https://github.com/horovod/horovod). Any questions or comments can be mailed to the first author.

### Requirements
The code has been tested to work with the following versions:
* python 3.6
* tensorflow-gpu 1.12
* horovod 0.16


### Datasets and preprocessing
**Training:** The MS1M dataset for training was downloaded from the InsightFace [repository](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo). The dataset is aligned by InsightFace authors using [MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html).

The downloaded dataset is in MXnet format. Use ``python preprocessing/msceleb_to_jpeg.py dataset_directory`` to extract jpeg images. Extracted images will be stored in ``dataset_directory/images``.

The MegaFace and FaceScrub datasets have been downloaded from the official MegaFace challenge [website](http://megaface.cs.washington.edu/participate/challenge.html).
