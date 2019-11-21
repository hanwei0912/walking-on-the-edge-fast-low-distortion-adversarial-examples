from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import scipy.io as si
import time
import tqdm

import torch
import torch.nn.functional as F
from torch.utils import data
from torch.optim import SGD, lr_scheduler
from torch.backends import cudnn

from torchvision import transforms
from torchvision.datasets import MNIST

from small_cnn import SmallCNN
import numpy as np


import pdb
import os
import argparse
parser = argparse.ArgumentParser(description='MNIST Training against DDN Attack')
parser.add_argument('--cpu', dest='cpu', action='store_true', help='force training on cpu')
args = parser.parse_args()
DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and not args.cpu) else 'cpu')
#test_set = MNIST(args.data, train=False, transform=transform, download=True)
#test_loader = data.DataLoader(test_set, batch_size=100, shuffle=True, num_workers=args.workers, pin_memory=True)
model = SmallCNN(drop=0.5).to(DEVICE)
model_dict = model.load_state_dict(torch.load('./model/adv_mnist.pth'))
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
###########

def result(X_test,Y_test,X_adv):
    L2_norm = np.sum((X_adv-X_test)**2,axis=(1,2,3))**.5
    X_test = torch.from_numpy(X_test)
    Y_test = torch.from_numpy(Y_test)
    X_adv = torch.from_numpy(X_adv)
    X_test,Y_test = X_test.to(DEVICE), Y_test.to(DEVICE)
    X_adv = X_adv.to(DEVICE)
    P=model(X_test.float()).argmax(1)
    acc = P.view(-1,1).float()==Y_test.float()
    P_adv=model(X_adv.float()).argmax(1)
    suc = P_adv.view(-1,1).float()==Y_test.float()
    pdb.set_trace()
    eval_params = {'batch_size': 128}
    accuracy = model_eval(sess, x_, y_, preds, X_adv, Y_test, args=eval_params)
    Y_adv = sess.run(preds,feed_dict={x_:X_adv,y_:Y_test})
    ## change mind rate
    ori_y = np.argmax(Y_test,axis=1)
    pre_y = np.argmax(Y_adv,axis=1)
    dif = np.abs(ori_y-pre_y)
    ind = np.ones(ori_y.shape)
    ind[np.nonzero(dif)] = 0

    p_test = sess.run(preds,feed_dict={x_:X_test,y_:Y_test})
    p_y = np.argmax(p_test,axis=1)
    dif = np.abs(ori_y-p_y)
    ind_o = np.ones(ori_y.shape)
    ind_o[np.nonzero(dif)] = 0

    dif = np.abs(pre_y-p_y)
    ind_c = np.ones(p_y.shape)
    ind_c[np.nonzero(dif)] = 0

    return accuracy,L2_norm,ind,ind_o,ind_c,L2_worst


eval_params ={'batch_size':128}

save_dir = "/nfs/nas4/data-hanwei/data-hanwei/DATA/Search/pytorch"
#file_name = "ori.mat"
#file_path = os.path.join(save_dir,file_name)
#data=si.loadmat(file_path)
#X_test = data['ori']
#Y_test = data['y']


file_name = "bp100.mat"
file_path = os.path.join(save_dir,file_name)
data=si.loadmat(file_path)
X_adv = data['adv']
X_test = data['ori']
Y_test = data['y']
pdb.set_trace()
acc, l2_n, ind, ind_o,ind_c,L2_w = result(X_test,Y_test,X_adv)
file_name = "process/fgsm-BasicCnn-"+eps+"-l2p.mat"
save_path = os.path.join(save_dir,file_name)
si.savemat(save_path,{'acc':acc,'l2':l2_n,'p':ind,'ori_a':ind_o,'c':ind_c,'l2_worst':L2_w})

