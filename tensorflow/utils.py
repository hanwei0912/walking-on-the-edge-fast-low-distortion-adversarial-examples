from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from abc import ABCMeta
from six.moves import xrange
import warnings
import collections
import math
import pdb

import cleverhans.utils as utils
from cleverhans.model import Model, CallableModelWrapper
from cleverhans.attacks import Attack
from cleverhans.compat import softmax_cross_entropy_with_logits
from cleverhans.compat import reduce_max, reduce_sum, reduce_min

import tensorflow as tf
import numpy as np

def quantization(x, levels):
    return tf.round(x *(levels - 1)) / (levels -1)

def is_adversarial(logits, y):
    pred_labels = tf.cast(tf.argmax(logits, 1),tf.int32)
    flag = tf.equal(pred_labels, y)
    flag = tf.logical_not(flag)
    return flag,pred_labels

def norm_l2(grad):
    avoid_zero_div = 1e-12
    reduc_ind = list(xrange(1, len(grad.get_shape())))
    a = tf.sqrt(tf.maximum(avoid_zero_div,
                reduce_sum(tf.square(grad),
                            reduc_ind,
                            keepdims=True)))
    return a

def norm_clip(grad, eps, ord):
    avoid_zero_div = 1e-12
    reduc_ind = list(xrange(1, len(grad.get_shape())))
    if ord == np.inf:
        eta = tf.clip_by_value(grad, -eps, eps)
    else:
        if ord == 1:
            norm = tf.maximum(avoid_zero_div,
              reduce_sum(tf.abs(grad),
                         reduc_ind, keepdims=True))
        elif ord == 2:
            # avoid_zero_div must go inside sqrt to avoid a divide by zero
            # in the gradient through this operation
            norm = tf.sqrt(tf.maximum(avoid_zero_div,
                                      reduce_sum(tf.square(grad),
                                                 reduc_ind,
                                                 keepdims=True)))
        # We must *clip* to within the norm ball, not *normalize* onto the
        # surface of the ball
        #factor = tf.minimum(tf.ones_like(norm), tf.div(tf.ones_like(norm)*eps, norm))
        #eta = grad * factor
        temp = tf.div(grad, norm)
        eps = tf.tile(tf.reshape(eps,(eps.shape[0],1,1,1)),
                [1,temp.shape[1],temp.shape[2],temp.shape[3]])
        eta = tf.multiply(temp, eps)
    return eta

def generate_grad_orth(d_x, grad):
    shape = d_x.get_shape().as_list()
    avoid_zero_div = 1e-12
    reduc_ind = list(xrange(1, len(grad.get_shape())))
    norm = tf.sqrt(tf.maximum(avoid_zero_div,
                              reduce_sum(tf.square(grad),
                                         reduc_ind,
                                         keepdims=True)))
    tpm = tf.reduce_sum(tf.multiply(d_x,grad),axis=[1,2,3])
    tpm = tf.reshape(tpm, norm.get_shape().as_list())
    l_t = tf.div(tpm, norm)
    r_t = tf.div(grad, norm)
    grad_orth = - d_x + l_t*r_t
    return grad_orth

def cosine_decay(global_step,decay_steps,alpha,learning_rate):
    global_step = tf.minimum(global_step, decay_steps)
    cosine_decay = 0.5 * (1 +
            tf.cos(tf.constant(math.pi)*tf.cast(global_step,tf.float32)/tf.cast(decay_steps,tf.float32)))
    decayed = (1-alpha)*cosine_decay + alpha
    eps = learning_rate * decayed
    return eps

def polynomial_decay(global_step,decay_steps,decay_rate,learning_rate,end_learning_rate):
    global_step = tf.minimum(global_step, decay_steps)
    rang = learning_rate-end_learning_rate
    radio = 1-(tf.cast(global_step,tf.float32)/tf.cast(decay_steps,tf.float32))
    eps = rang*tf.pow(radio,tf.cast(decay_rate,tf.float32))+end_learning_rate
    return eps

def teddy_decay(global_step,decay_steps,min_learning_rate):
    global_step = tf.minimum(global_step, decay_steps)
    rate = tf.cast(global_step,tf.float32)/tf.cast(decay_steps+1,tf.float32)
    rage = 1 -  min_learning_rate
    eps = min_learning_rate + tf.multiply(rate, rage)
    return eps

def softmax_cross_entropy_better(logits, y_hot):
    tmp = y_hot * logits
    logits_1 = logits - tmp
    j_best = tf.reduce_max(logits_1,axis=1)
    j_best_v = tf.tile(tf.reshape(j_best,(j_best.shape[0],1)),(1,y_hot.shape[1]))
    logits_2 = logits_1 - j_best_v + y_hot*j_best_v
    tmp_s = tf.reduce_max(tmp, axis=1)
    up = tmp_s - j_best
    down = tf.log(tf.reduce_sum(tf.exp(logits_2)+1,axis=1))
    loss = up - down
    return loss

def floor_quant(x, levels):
    return tf.floor(x*(levels - 1)) / (levels - 1)

def ceil_quant(x, levels):
    return tf.ceil(x*(levels-1)) / (levels-1)

def psi(ng,nd):
    cos_psi = tf.reduce_sum(tf.multiply(nd,ng),axis=[1,2,3])
    sin_psi = tf.sqrt(1-tf.square(cos_psi))

    sin_psi = tf.tile(tf.reshape(sin_psi,(sin_psi.shape[0],1,1,1)),(1,nd.shape[1],nd.shape[2],nd.shape[3]))
    cos_psi = tf.tile(tf.reshape(cos_psi,(sin_psi.shape[0],1,1,1)),(1,nd.shape[1],nd.shape[2],nd.shape[3]))
    return sin_psi, cos_psi

def psi_old(ng,nd):
    cos_psi = tf.reduce_sum(tf.multiply(nd,ng),axis=[1,2,3])
    sin_psi = tf.sqrt(1-tf.square(cos_psi))
    tan_psi = tf.div(sin_psi,cos_psi)

    sin_psi = tf.tile(tf.reshape(sin_psi,(sin_psi.shape[0],1,1,1)),(1,nd.shape[1],nd.shape[2],nd.shape[3]))
    tan_psi = tf.tile(tf.reshape(tan_psi,(sin_psi.shape[0],1,1,1)),(1,nd.shape[1],nd.shape[2],nd.shape[3]))
    return tan_psi, sin_psi

def out_direction(d,g,ng,sng):
    lambd = tf.reduce_sum(tf.div(tf.multiply(d,ng),sng),axis=[1,2,3])
    lambd = tf.tile(tf.reshape(lambd,(lambd.shape[0],1,1,1)),(1,d.shape[1],d.shape[2],d.shape[3]))
    g_ort = d - tf.multiply(lambd,g)
    return g_ort

def out_p(d,g,ng,snd,sng,sin_psi,beta,g_ort):
    #lambd = tf.reduce_sum(tf.div(tf.multiply(d,ng),sng),axis=[1,2,3])
    #lambd = tf.tile(tf.reshape(lambd,(lambd.shape[0],1,1,1)),(1,d.shape[1],d.shape[2],d.shape[3]))
    #beta = tf.tile(tf.reshape(beta,(beta.shape[0],1,1,1)),(1,d.shape[1],d.shape[2],d.shape[3]))
    mu = tf.multiply(tf.div(snd,beta), sin_psi) - 1
    flag = tf.less(mu,tf.zeros_like(mu))
    mu = tf.where(flag,tf.zeros_like(mu),mu)
    #p = tf.div(d-tf.multiply(lambd,g),1+mu)
    p = tf.div(g_ort,1+mu)
    return p

def in_p(d,nd,ng,beta,sin_psi,tan_psi,snd,sng,dt):
    p = d - tf.multiply(ng,dt)
    snp = norm_l2(p)
    flag = tf.less(snp,tf.ones_like(snp)*beta)

    cos_phi = tf.square(beta)-tf.square(dt) + tf.square(snd)
    cos_phi = tf.div(cos_phi,(2*tf.multiply(snd,beta)))
    sin_phi = tf.sqrt(1-tf.square(cos_phi))
    p_o = ((cos_phi+tf.div(sin_phi,tan_psi))*nd - tf.div(sin_phi,sin_psi)*ng)*beta
 
    p = tf.where(flag,p,p_o)
    return p

def Mdistortion(p,levels):
    pq = quantization(p, levels)
    dis_c = tf.sqrt(tf.reduce_sum(tf.square(pq),axis=[1,2,3]))
    dis_c = tf.tile(tf.reshape(dis_c,[dis_c.shape[0],1,1,1]),(1,p.shape[1],p.shape[2],p.shape[3]))
    return dis_c

def estimate_beta_in_simple(d,g,snd,dis):
    ngo = tf.tile(norm_l2(g),(1,d.shape[1],d.shape[2],d.shape[3]))
    g_ort = tf.div(g,ngo)
    p_o =tf.tile(tf.reshape(tf.reduce_sum(tf.multiply(d,g_ort),axis=[1,2,3]),
        (d.shape[0],1,1,1)),(1,d.shape[1],d.shape[2],d.shape[3]))
    bac = tf.square(dis) - tf.square(snd) + tf.square(p_o)
    beta = p_o + tf.sqrt(bac)
    #beta = 0.17 + 0.426*tf.pow(beta,0.4)
    beta_min = 0.1*tf.ones_like(dis)
    beta_max = dis -snd
    #beta_max = 0.17 + 0.426*tf.pow(beta_max,0.4)
    #flag = tf.less(beta,beta_max)
    #beta = tf.where(flag,beta,beta_max)
    flag = tf.less(beta,beta_min)
    beta = tf.where(flag,beta_min,beta)
    return beta, g_ort

def estimate_beta_out_LS(d,g_ort,dis,num_step,levels):
    ngo = tf.tile(norm_l2(g_ort),(1,d.shape[1],d.shape[2],d.shape[3]))
    g_ort = tf.div(g_ort,ngo)
    tmp =tf.tile(tf.reshape(tf.reduce_sum(tf.multiply(d,g_ort),axis=[1,2,3]),
        (d.shape[0],1,1,1)),(1,d.shape[1],d.shape[2],d.shape[3]))
    an_tmp = tf.square(tmp) - tf.square(d) + tf.square(dis)
    min_beta = (tmp-tf.sqrt(an_tmp))
    max_beta = (tmp)

    p_min = tf.multiply(min_beta,g_ort)
    p_max = tf.multiply(max_beta,g_ort)

    DMin = Mdistortion(d-p_min,levels)
    DMax = Mdistortion(d-p_max,levels)

    def cond(i,min_beta,max_beta,DMin,DMax):
        return tf.less(i,num_step)

    def body(i,min_beta,max_beta,DMin,DMax):
        beta = (min_beta + max_beta)/2
        p = tf.multiply(beta,g_ort)
        D = Mdistortion(d-p,levels)

        flag = tf.less(D,dis)
        DMin = tf.where(flag,D,DMin)
        DMax = tf.where(flag,DMax,D)

        min_beta = tf.where(flag, beta, min_beta)
        max_beta = tf.where(flag, max_beta, beta)

        return i+1, min_beta,max_beta,DMin,DMax

    _, min_beta,max_beta,DMin,DMax = tf.while_loop(cond,body,[tf.zeros([]),
        max_beta,min_beta,DMin,DMax])
    dMax = tf.abs(DMax-dis)
    dMin = tf.abs(DMin-dis)
    flag = tf.less(dMax,dMin)
    beta = tf.where(flag,max_beta,min_beta)
    return beta
