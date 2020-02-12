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

if (( $# != 1 )); then
  echo "Usage: preprocess_msceleb.sh [data dir]"
  exit
fi

# Build the TFRecords version of the ImageNet data.
echo "Now creating TFRecord files. This might take a while."
BUILD_SCRIPT="build_image_data.py"

IMAGES_DIR="${1}/images"
OUTPUT_DIR="${1}/tfrecords/train"
LABELS_FILE="${1}/labels.txt"

echo "Writing to ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

python "${BUILD_SCRIPT}" \
  --train_directory="${IMAGES_DIR}" \
  --output_directory="${OUTPUT_DIR}" \
  --labels_file="${LABELS_FILE}"

echo "Done"
