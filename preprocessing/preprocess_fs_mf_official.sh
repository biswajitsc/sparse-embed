#!/bin/bash

# Script to preprocess a set of labelled images and store in
# sharded TFRecord files. The sharded files are stored under
# the data directory.
# Assumes that the images are organized into folders according
# to their labels inside the data directory.
# synsets.txt should contain a list of the labels. For example
#   n03114236
#   n03114807
#   ...
#
# usage:
#  ./preprocess_image_data.sh [data-dir]

set -e

if (( $# != 2 )); then
  echo "Usage: preprocess_[dataset].sh [data dir] [file list]"
  exit
fi

# Build the TFRecords version of the ImageNet data.
echo "Now creating TFRecord files. This might take a while."
BUILD_SCRIPT="build_image_data.py"

IMAGES_DIR="${1}/aligned_if/"
OUTPUT_DIR="${1}/tfrecords_official/"
FILELIST="${2}"

echo "Writing to ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=""
python "${BUILD_SCRIPT}" \
  --validation_directory="${IMAGES_DIR}" \
  --output_directory="${OUTPUT_DIR}" \
  --filelist="${FILELIST}" \
  --validation_shards=128

echo "Done"
