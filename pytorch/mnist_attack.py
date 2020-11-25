import os
import argparse
import tqdm
import pdb
import scipy.io as si
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils import data
from torch.optim import SGD, lr_scheduler
from torch.backends import cudnn

from torchvision import transforms
from torchvision.datasets import MNIST

from small_cnn import SmallCNN
from utils import AverageMeter, save_checkpoint, requires_grad_
from bp import BP
from ddn import DDN
from pgd import PGD
from fgsm import FGSM
from ifgsm import IFGSM

parser = argparse.ArgumentParser(description='MNIST Training against DDN Attack')

parser.add_argument('--data', default='data/mnist', help='path to dataset')
parser.add_argument('--workers', default=2, type=int, help='number of data loading workers')
parser.add_argument('--cpu', dest='cpu', action='store_true', help='force training on cpu')

args = parser.parse_args()
print(args)

DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and not args.cpu) else 'cpu')

transform = transforms.Compose([transforms.ToTensor()])

test_set = MNIST(args.data, train=False, transform=transform, download=True)
test_loader = data.DataLoader(test_set, batch_size=100, shuffle=True, num_workers=args.workers, pin_memory=True)

model = SmallCNN().to(DEVICE)
model_dict = model.load_state_dict(torch.load('./model/mnist.pth'))

#attacker = FGSM(eps=0.3, device=DEVICE)
#attacker = IFGSM(steps=20,eps=0.3,eps_iter=0.1, device=DEVICE)
#attacker = PGD(steps=20,eps=4,eps_iter=3, device=DEVICE)
attacker = BP(steps=20, device=DEVICE, gamma=0.7)
#attacker = DDN(steps=100, device=DEVICE)

requires_grad_(model, True)
model.eval()

ori_image = np.zeros((10000,1,28,28))
ori_label = np.zeros((10000,1))
adv_image = np.zeros((10000,1,28,28))
adv_label = np.zeros((10000,1))

#requires_grad_(model, False)
for i, (images, labels) in enumerate(tqdm.tqdm(test_loader, ncols=80)):
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    #best_x = attacker.attack(model, images, labels)
    adv,best_x = attacker.attack(model, images, labels)
    adv_pred = model(best_x).argmax(1)
    i_sta = i*100
    i_end = (i+1)*100
    ori_label[i_sta:i_end]=labels.cpu().detach().numpy().reshape((100,1))
    ori_image[i_sta:i_end]=images.cpu().detach().numpy()
    adv_image[i_sta:i_end]=best_x.cpu().detach().numpy()
    adv_label[i_sta:i_end]=adv_pred.cpu().detach().numpy().reshape((100,1))

# Compute
success_rate = np.mean(adv_label != ori_label)
norms = np.linalg.norm((adv_image - ori_image).reshape(ori_image.shape[0], -1), axis=1)
mean_l2 = np.mean(norms)
median_l2 = np.median(norms)

print('Attack results - ASR: {:.3%} - Mean L2: {:.4f} - Median L2: {:.4f}'.format(success_rate,
mean_l2, median_l2))
path_save = '/nfs/nas4/data-hanwei/data-hanwei/DATA/Search/pytorch/ifgsm-test.mat'
si.savemat(path_save,{'adv':adv_image,'ori':ori_image,'y':ori_label,'pred':adv_label})

