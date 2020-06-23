from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import pdb
import time

from attacks import KKTFun5
from cleverhans.utils_tf import model_train, model_eval, tf_model_load
import numpy as np
from PIL import Image
from cleverhans.model import Model
import scipy.io as si
from basic_cnn_models import InceptionModel, _top_1_accuracy
from load_data import load_images, save_images

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim

FLAGS = tf.flags.FLAGS

def main(_):
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    y_label = tf.placeholder(tf.int32, shape=(FLAGS.batch_size,))
    y_hot   = tf.one_hot(y_label, num_classes)

    model = InceptionModel(num_classes)
    preds = model(x_input)
    logits = model.get_logits(x_input)
    acc = _top_1_accuracy(logits, y_label)
    tf_model_load(sess, FLAGS.checkpoint_path)

    attack = KKTFun5(model, sess=sess)
    eps = FLAGS.eps
    alp = FLAGS.alp
    params = {'eps':0.3,
              'alp':1.0,
              'ord':2,
              'nb_iter':eps,
              'clip_min':0.,
              'clip_max':1.}
    adv_x, log_step, log_suc = attack.generate(x_input, y_label, **params)
    adv_image = np.zeros((1000,299,299,3))
    l2_norm = np.zeros((1000))
    acc_ori = np.zeros((1000))
    acc_val = np.zeros((1000))
    pred_score = np.zeros((1000,1001))
    pred_score_adv = np.zeros((1000,1001))
    name = []
    b_i = 0

    begin = time.time()
    for images, _, labels, filenames in load_images(FLAGS.input_dir, FLAGS.input_dir, FLAGS.metadata_file_path, batch_shape):
        bb_i = b_i + FLAGS.batch_size
        y_labels = np.zeros((FLAGS.batch_size,num_classes))
        for i_y in range(FLAGS.batch_size):
            y_labels[i_y][labels[i_y]]=1
        x_adv = sess.run(adv_x,feed_dict={x_input:images,y_label:labels})
        #acc_val[b_i:bb_i] = sess.run(log_suc,feed_dict={x_input:images,y_label:labels})
        #l2_norm[b_i:bb_i] = sess.run(log_step,feed_dict={x_input:images,y_label:labels})
        #pdb.set_trace()
        #acc_ori[b_i] = sess.run(acc,feed_dict={x_input:images,y_label:labels})
        #l2_norm[b_i] = np.mean(np.sum((images- x_adv)**2,axis=(1,2,3))**.5)
        adv_image[b_i:bb_i] = x_adv
        acc_ori[b_i:bb_i] = sess.run(acc, feed_dict={x_input:images,y_label:labels})
        acc_val[b_i:bb_i] = sess.run(acc, feed_dict={x_input:x_adv,y_label:labels})
        pred_score[b_i:bb_i] = sess.run(preds,feed_dict={x_input:images,y_label:labels})
        pred_score_adv[b_i:bb_i] = sess.run(preds,feed_dict={x_input:x_adv,y_label:labels})
        l2_norm[b_i:bb_i] = np.sum((x_adv-images)**2,axis=(1,2,3))**.5
        name.append(filenames)
        b_i = bb_i

        #save_images(x_adv, filenames, FLAGS.output_dir)
    #print(time.time()-begin)
    #path_data = '/nfs/nas4/data-hanwei/data-hanwei/DATA/Search/inception/Ens4/OUR/'+str(eps)+'.mat'
    #si.savemat(path_data,{'x_adv':adv_image,'ori_a':acc_ori,'p':acc_val,'l2':l2_norm,'name':name,'pred_o':pred_score,'pred_a':pred_score_adv})
    #path_save='/nfs/nas4/data-hanwei/data-hanwei/DATA/Search/inception/OUR1/'+str(eps)+'.mat'
    #si.savemat(path_save,{'suc':acc_val,'norm':l2_norm})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("eps",help = "step size")
    parser.add_argument("alp",help = "step size")
    args = parser.parse_args()
    tf.flags.DEFINE_integer(
        'eps', args.eps, 'Step size')
    tf.flags.DEFINE_float(
        'alp', args.alp, 'Step size')
    tf.flags.DEFINE_string(
        'master', '', 'The address of the TensorFlow master to use.')
    tf.flags.DEFINE_string(
        'checkpoint_path',
        '/nfs/pyrex/raid6/hzhang/2017-nips/models/ens_adv_inception/ens4/ens4_adv_inception_v3.ckpt', 'Path to checkpoint for inception network.')
        #'checkpoint_path', '/nfs/pyrex/raid6/hzhang/2017-nips/models/inception_v3.ckpt', 'Path to checkpoint for inception network.')
    tf.flags.DEFINE_string(
        'input_dir', '/nfs/pyrex/raid6/hzhang/2017-nips/images', 'Input directory with images.')
    path_save = '/nfs/nas4/data-hanwei/data-hanwei/DATA/Search/inception/Ens4/OUR/'+str(args.eps)
    #path_save = '/nfs/nas4/data-hanwei/data-hanwei/DATA/Search/inception/OUR/BetterGamma/'+str(args.eps)
    folder = os.path.exists(path_save)
    if not folder:
        os.makedirs(path_save)
    tf.flags.DEFINE_string(
        'output_dir', path_save, 'Output directory with images.')
    tf.flags.DEFINE_float(
        'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')
    tf.flags.DEFINE_integer(
        'image_width', 299, 'Width of each input images.')
    tf.flags.DEFINE_integer(
        'image_height', 299, 'Height of each input images.')
    tf.flags.DEFINE_integer(
        'batch_size', 20, 'How many images process at one time.')
    tf.flags.DEFINE_string(
        'metadata_file_path',
        '/nfs/pyrex/raid6/hzhang/2017-nips/dev_dataset.csv',
        'Path to metadata file.')
    tf.app.run()
