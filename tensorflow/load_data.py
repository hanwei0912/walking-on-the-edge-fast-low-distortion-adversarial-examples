from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import pdb
import numpy as np
from PIL import Image
import scipy.io as si
import tensorflow as tf

def load_images(input_dir, ori_input_dir, metadata_file_path,  batch_shape):
    """Read png images from input directory in batches.
    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array,
      i.e. [batch_size, height, width, 3]
    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    with open(metadata_file_path) as input_file:
        reader = csv.reader(input_file)
        header_row = next(reader)
        rows = list(reader)

    images = np.zeros(batch_shape)
    X_test = np.zeros(batch_shape)
    labels = np.zeros(batch_shape[0], dtype=np.int32)
    rows = np.array(rows)
    row_idx_true_label = header_row.index('TrueLabel')
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath) as f:
            image = np.array(Image.open(f).convert('RGB')
                             ).astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image
        #images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        ori_filepath = os.path.join(ori_input_dir,os.path.basename(filepath))
        with tf.gfile.Open(ori_filepath) as f:
            ori_image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
        X_test[idx,:,:,:]=ori_image
        #X_test[idx,:,:,:]=ori_image * 2.0 - 1.0
        (name,_) = os.path.splitext(os.path.basename(filepath))
        (ind_l,_) = np.where(rows==name)
        row = rows[ind_l][0]
        labels[idx] = int(row[row_idx_true_label])
        idx += 1
        if idx == batch_size:
            yield images, X_test, labels, filenames
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield images, X_test, labels, filenames

def save_images(images, filenames, output_dir):
  """Saves images to the output directory.
  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
      img = (images[i, :, :, :] * 255.0).astype(np.uint8)
      #img = (np.round(images[i, :, :, :] * 255.0)).astype(np.uint8)
      #img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
      Image.fromarray(img).save(f, format='PNG')
