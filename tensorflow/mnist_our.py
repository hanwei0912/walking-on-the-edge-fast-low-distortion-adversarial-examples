from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import numpy as np
import tensorflow as tf
import pdb
import scipy.io as si

from attacks import KKTFun5
from cleverhans.dataset import MNIST
from tensorflow.python.platform import flags
from cleverhans.loss import CrossEntropy
from cleverhans.utils import grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_eval, tf_model_load
from cleverhans.train import train
from basic_cnn_models import ModelBasicCNN

FLAGS = flags.FLAGS

VIZ_ENABLED = True
BATCH_SIZE = 128
NB_EPOCHS = 6
SOURCE_SAMPLES = 10
LEARNING_RATE = .001
CW_LEARNING_RATE = .2
ATTACK_ITERATIONS = 100
MODEL_PATH = os.path.join('models', 'mnist')
TARGETED = True


def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, viz_enabled=VIZ_ENABLED,
                   nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                   source_samples=SOURCE_SAMPLES,
                   learning_rate=LEARNING_RATE,
                   attack_iterations=ATTACK_ITERATIONS,
                   model_path=MODEL_PATH,
                   targeted=TARGETED):
  """
  MNIST tutorial for Carlini and Wagner's attack
  :param train_start: index of first training set example
  :param train_end: index of last training set example
  :param test_start: index of first test set example
  :param test_end: index of last test set example
  :param viz_enabled: (boolean) activate plots of adversarial examples
  :param nb_epochs: number of epochs to train model
  :param batch_size: size of training batches
  :param nb_classes: number of output classes
  :param source_samples: number of test inputs to attack
  :param learning_rate: learning rate for training
  :param model_path: path to the model file
  :param targeted: should we run a targeted attack? or untargeted?
  :return: an AccuracyReport object
  """
  # Object used to keep track of (and return) key accuracies
  report = AccuracyReport()

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  # Create TF session
  sess = tf.Session()
  print("Created TensorFlow session.")

  set_log_level(logging.DEBUG)

  # Get MNIST test data
  mnist = MNIST(train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end)
  x_train, y_train = mnist.get_set('train')
  x_test, y_test = mnist.get_set('test')
  x_val = x_train[0:10000]
  y_val = y_train[0:10000]

  # Obtain Image Parameters
  img_rows, img_cols, nchannels = x_train.shape[1:4]
  nb_classes = y_train.shape[1]

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                        nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))
  nb_filters = 64

  # Define TF model graph
  model = ModelBasicCNN('model1', nb_classes, nb_filters)
  preds = model.get_logits(x)
  loss = CrossEntropy(model, smoothing=0.1)
  print("Defined TensorFlow model graph.")

  ###########################################################################
  # Training the model using TensorFlow
  ###########################################################################

  # Train an MNIST model
  train_params = {
      'nb_epochs': nb_epochs,
      'batch_size': batch_size,
      'learning_rate': learning_rate,
      'filename': os.path.split(model_path)[-1]
  }

  rng = np.random.RandomState([2017, 8, 30])
  # check if we've trained before, and if we have, use that pre-trained model
  if os.path.exists(model_path + ".meta"):
    tf_model_load(sess, model_path)
  else:
    train(sess, loss, x_train, y_train, args=train_params, rng=rng)
    saver = tf.train.Saver()
    saver.save(sess, model_path)

  # Evaluate the accuracy of the MNIST model on legitimate test examples
  eval_params = {'batch_size': batch_size}
  accuracy = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
  assert x_test.shape[0] == test_end - test_start, x_test.shape
  print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
  report.clean_train_clean_eval = accuracy

  ###########################################################################
  # Craft adversarial examples using Carlini and Wagner's approach
  ###########################################################################
  attack = KKTFun5(model, sess=sess)
  eps = FLAGS.eps
  alp = FLAGS.alp
  params = {'eps':0.7,
            'alp':1,
            'ord':2,
            'nb_iter':eps,
            'clip_min':0.,
            'clip_max':1.}
  x_ = tf.reshape(x,(10000, img_rows, img_cols, nchannels))
  y_ = tf.argmax(tf.reshape(y,(10000,10)),axis=1)
  y_ = tf.cast(y_, tf.int32)
  adv_x,suc, l2log = attack.generate(x_, y_, **params)
  #adv_image = np.zeros_like(x_train)
  #for i in range(6):
  #    i_b = i*10000
  #    i_e = (i+1)*10000
  #    adv_image[i_b:i_e] = sess.run(adv_x, feed_dict={x:x_train[i_b:i_e],y:y_train[i_b:i_e]})
  #succ = sess.run(suc, feed_dict={x:x_test,y:y_test})
  adv_image = sess.run(adv_x, feed_dict={x:x_val,y:y_val})
  l2 = np.sum((adv_image-x_val)**2,axis=(1,2,3))**.5
  adv_pred = sess.run(preds,feed_dict={x:adv_image,y:y_val})
  ori_y = np.argmax(y_val,axis=1)
  pre_y = np.argmax(adv_pred,axis=1)
  dif = np.abs(ori_y-pre_y)
  ind = np.ones(y_val.shape)
  ind[np.nonzero(dif)] = 0
  p_test = sess.run(preds,feed_dict={x:x_val,y:y_val})
  p_y = np.argmax(p_test,axis=1)
  dif = np.abs(ori_y-p_y)
  ind_o = np.ones(ori_y.shape)
  ind_o[np.nonzero(dif)] = 0
  #path_save='/nfs/nas4/data-hanwei/data-hanwei/DATA/Search/mnist/adv_new/suc-our-BasicCnn'+str(eps)+'.mat'
  path_save ='/nfs/nas4/data-hanwei/data-hanwei/DATA/Search/mnist/adv/fgsm/our-BasicCnn-'+str(eps)+'.mat'
  si.savemat(path_save,{'img':adv_image,'l2':l2,'p':ind,'ori_a':ind_o})
  print(l2)

  #eval_params = {'batch_size': 100}
  #err = model_eval(sess, x, y, preds, adv_x, y_test,
  #                 args=eval_params)
  #adv_accuracy = 1 - err

  ## Compute the number of adversarial examples that were successfully found
  #print('Avg. rate of successful adv. examples {0:.4f}'.format(adv_accuracy))
  #report.clean_train_adv_eval = 1. - adv_accuracy

  ## Compute the average distortion introduced by the algorithm
  #percent_perturbed = np.mean(np.sum((adv - adv_inputs)**2,
  #                                   axis=(1, 2, 3))**.5)
  #print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))

  # Close TF session
  sess.close()

  # Finally, block & display a grid of all the adversarial examples

  return report


def main(argv=None):
  from cleverhans_tutorials import check_installation
  check_installation(__file__)

  mnist_tutorial(viz_enabled=FLAGS.viz_enabled,
                    nb_epochs=FLAGS.nb_epochs,
                    batch_size=FLAGS.batch_size,
                    source_samples=FLAGS.source_samples,
                    learning_rate=FLAGS.learning_rate,
                    attack_iterations=FLAGS.attack_iterations,
                    model_path=FLAGS.model_path,
                    targeted=FLAGS.targeted)


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
  flags.DEFINE_boolean('viz_enabled', VIZ_ENABLED,
                       'Visualize adversarial ex.')
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                       'Number of epochs to train model')
  flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Size of training batches')
  flags.DEFINE_integer('source_samples', SOURCE_SAMPLES,
                       'Number of test inputs to attack')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')
  flags.DEFINE_string('model_path', MODEL_PATH,
                      'Path to save or load the model file')
  flags.DEFINE_integer('attack_iterations', ATTACK_ITERATIONS,
                       'Number of iterations to run attack; 1000 is good')
  flags.DEFINE_boolean('targeted', TARGETED,
                       'Run the tutorial in targeted mode?')

  tf.app.run()
