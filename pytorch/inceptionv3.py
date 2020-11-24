import os
import argparse
import time
import tqdm
import pdb
import random
import os
import numpy as np
import sklearn.preprocessing as processing


import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torchvision.datasets as datasets

from torch.utils import data
from torch.optim import SGD, lr_scheduler
from torch.backends import cudnn

from torchvision import transforms
import torchvision
#from torchvision.datasets import ImageNet
import torchvision.models as models

from utils import *
from bp import BP

valdir = os.path.join('/nfs/nas4/data-hanwei/data-hanwei/DATA', 'ILSVRC2012_val')
seed = 1
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
GLOBAL_SEED = 1
 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
GLOBAL_WORKER_ID = None
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #normalize,
    ])),
    batch_size=20, shuffle=True,
    num_workers=2, pin_memory=True,  worker_init_fn=worker_init_fn)

DEVICE = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
attacker = BP(steps=20, device=DEVICE)

inception = models.inception_v3(pretrained=True)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
preprocess_layer = Preprocessing_Layer(mean,std)
model = nn.Sequential(preprocess_layer, inception)
model.cuda()
model.eval()
top1 = AverageMeter('Acc@1', ':6.2f')
adv1 = AverageMeter('Adv@1', ':6.2f')
Norml2 = AverageMeter('Pnorml2@1', ':6.2f')
progress = ProgressMeter(
        len(val_loader),
        [top1,adv1,Norml2],
        prefix='Test: ')
begin = time.time()
for i, (images, target) in enumerate(val_loader):
    images = images.cuda()
    target = target.cuda()
    currnt_adv, adv = attacker.attack(model, images, target)
    norm = np.sum((images.data.cpu().detach().numpy()- adv.data.cpu().detach().numpy())**2,axis=(1,2,3))**.5
    Norml2.update(np.mean(norm),images.size(0))
    torch.cuda.empty_cache()
    output = model(images)
    output_adv = model(adv)
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    acc11,acc55 = accuracy(output_adv,target,topk=(1,5))
    top1.update(acc1.cpu().detach().numpy()[0], images.size(0))
    adv1.update(acc11.cpu().detach().numpy()[0], images.size(0))
    del acc11,acc1,output,output_adv,acc5,acc55
    torch.cuda.empty_cache()
    progress.display(i)
print(time.time()-begin)
