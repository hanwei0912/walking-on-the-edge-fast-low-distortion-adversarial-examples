import os
import argparse
import tqdm
import pdb

import torch
import torch.nn.functional as F
from torch.utils import data
from torch.optim import SGD, lr_scheduler
from torch.backends import cudnn

from torchvision import transforms
import torchvision
#from torchvision.datasets import ImageNet
import torchvision.models as models

from utils import AverageMeter, save_checkpoint, requires_grad_

pdb.set_trace()
imagenet_data = torchvision.datasets.ImageNet('/nfs/pyrex/raid6/hzhang/2017-nips/images/')
data_loader = torch.utils.data.DataLoader(imagenet_data,batch_size=4,shuffle=True,num_workers=2)

inception = models.inception_v3(pretrained=True)
inception.eval()

pdb.set_trace()
