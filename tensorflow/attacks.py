from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from abc import ABCMeta
from six.moves import xrange
import warnings
import collections
import pdb

import cleverhans.utils as utils
import math
from cleverhans.model import Model, CallableModelWrapper
from cleverhans.attacks import Attack
from cleverhans.compat import softmax_cross_entropy_with_logits
from cleverhans.compat import reduce_max, reduce_sum
from utils import *

import tensorflow as tf
import numpy as np


class KKTFun5(Attack):
    def __init__(self, model, sess=None, dtypestr= 'float32', **kwargs):
        super(KKTFun5, self).__init__(model, sess, dtypestr, **kwargs)

        self.feedable_kwargs = ('alp','eps','y','y_target','clip_min','clip_max')
        self.structural_kwargs = ['ord','nb_iter']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'probs')

    def generate(self, x, y, **kwargs):
        self.parse_params(**kwargs)
        flag_cross = tf.cast(tf.zeros(x.shape[0],), tf.bool)

        def cond(i, adv_x, best_x, flag_cross, log_step, log_suc):
            return tf.less(i,self.nb_iter)

        def body(i, adv_x, best_x, flag_cross, log_step, log_suc):
            targeted = (self.y_target is not None)
            logits = self.model.get_logits(adv_x)
            y_hot = tf.one_hot(y,logits.shape[1])
            loss = softmax_cross_entropy_better(logits, y_hot)
            grad, = tf.gradients(loss,adv_x)
            g = - grad

            d = x - adv_x
            snd = tf.tile(norm_l2(d),(1,d.shape[1],d.shape[2],d.shape[3]))
            sng = tf.tile(norm_l2(g),(1,g.shape[1],g.shape[2],g.shape[3]))
            nd = tf.div(d,snd)
            ng = tf.div(g,sng)
            tan_psi, sin_psi = psi_old(ng,nd)

            # cosine decay
            self.eps_all = teddy_decay(i, tf.ones(x.shape[0])*self.nb_iter, self.eps)
            epsi = tf.tile(tf.reshape(self.eps_all,(d.shape[0],1,1,1)),(1,d.shape[1],d.shape[2],d.shape[3]))
            alpe = tf.multiply(self.alp*tf.ones_like(epsi),epsi)

            # search
            p_search = tf.multiply(ng,alpe)

            # refine step
            # out
            g_ort = out_direction(d,g,ng,sng)
            epsi_out = tf.multiply(snd,epsi)
            beta = estimate_beta_out_LS(d,g_ort,tf.multiply(snd,epsi),7,self.levels)
            p_out = out_p(d,g,ng,snd,sng,sin_psi,beta,g_ort)
            beta,nor = estimate_beta_in_simple(d,g,snd,tf.div(snd,epsi))
            p_in = tf.multiply(ng,beta)

            flag,pred_l1 = is_adversarial(logits, y)

            flag_cross =tf.logical_or(flag_cross, flag)
            delta = tf.where(flag_cross, p_in, p_search)
            delta = tf.where(flag, p_out, delta)

            adv_x = tf.clip_by_value(adv_x + delta, self.clip_min, self.clip_max)
            # Quantization
            adv_x = quantization(adv_x, self.levels)
            logits = self.model.get_logits(adv_x)
            flag, pred_l = is_adversarial(logits,y)

            # save best
            a = norm_l2(best_x -x)
            b = norm_l2(adv_x -x)

            flag_save = tf.reshape(tf.greater(a,b),(x.shape[0],))
            nm_best_x = tf.where(flag_save, adv_x, best_x)
            best_x = tf.where(flag, nm_best_x, best_x)

            # save log
            tmp = tf.one_hot(tf.cast(i, tf.int32),self.nb_iter)
            log_step = log_step + tf.reshape(norm_l2(adv_x-x),(delta.shape[0],1))*tmp
            log_suc = log_suc + tf.reshape(tf.cast(flag, tf.float32),(flag.shape[0],1))*tmp

            return i+1, adv_x, best_x, flag_cross, log_step, log_suc

        adv_x = x
        best_x = tf.ones_like(x)
        log_step = tf.zeros([x.shape[0],self.nb_iter],tf.float32)
        log_suc = tf.zeros([x.shape[0],self.nb_iter],tf.float32)
        _, adv_x, best_x, flag_cross, log_step,log_suc = tf.while_loop(cond, body, [tf.zeros([]), adv_x,
            best_x, flag_cross, log_step, log_suc])
        return best_x, log_step, log_suc

    def parse_params(self, eps=0.3, alp=0.4, nb_iter=10, y=None, y_target=None, clip_min=None,
            clip_max=None, ord=2, levels = 256, **kwargs):

        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param eps: (optional float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (optional float) step size for each attack iteration
        :param nb_iter: (optional int) Number of attack iterations.
        :param y: (optional) A tensor with the true labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param decay_factor: (optional) Decay factor for the momentum term.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Save attack-specific parameters
        self.nb_iter = nb_iter
        self.eps = eps
        self.alp = alp
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.levels = levels

        if self.y is not None and self.y_target is not None:
          raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
          raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True

