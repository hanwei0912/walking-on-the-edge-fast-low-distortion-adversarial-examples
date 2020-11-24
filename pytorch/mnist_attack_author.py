import os
import argparse
import tqdm
import pdb
import scipy.io as si
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import MNIST

from small_cnn import SmallCNN
from utils import requires_grad_

from ddn import DDN
from bp import BP

parser = argparse.ArgumentParser(description='DDN Attack on MNIST')

parser.add_argument('--data', default='data/mnist', help='path to dataset')
parser.add_argument('--workers', default=2, type=int, help='number of data loading workers')
parser.add_argument('--cpu', dest='cpu', action='store_true', help='force training on cpu')

args = parser.parse_args()
print(args)

DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and not args.cpu) else 'cpu')

transform = transforms.ToTensor()
test_set = MNIST(args.data, train=False, transform=transform, download=True)
test_loader = data.DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=args.workers, pin_memory=True)

model = SmallCNN().to(DEVICE)
model_dict = model.load_state_dict(torch.load('model/adv_mnist.pth'))

attacker = BP(steps=100, device=DEVICE)
#attacker = DDN(steps=1000, device=DEVICE)

ori_image = []
ori_label = []
adv_image = []
adv_preds = []

requires_grad_(model, False)
model.eval()

for i, (images, labels) in enumerate(tqdm.tqdm(test_loader, ncols=80)):
    t_images, t_labels = images.to(DEVICE), labels.to(DEVICE)

    cadv,adv = attacker.attack(model, t_images, t_labels)
    adv_pred = model(adv).argmax(1)

    ori_image.append(images)
    ori_label.append(labels)
    adv_image.append(adv.cpu())
    adv_preds.append(adv_pred.cpu())

ori_image = torch.cat(ori_image, 0).numpy()
ori_label = torch.cat(ori_label, 0).numpy()
adv_image = torch.cat(adv_image, 0).numpy()
adv_preds = torch.cat(adv_preds, 0).numpy()

# Compute metrics with numpy as PyTorch had some problems with sums on large tensors
success_rate = np.mean(adv_preds != ori_label)
norms = np.linalg.norm((adv_image - ori_image).reshape(ori_image.shape[0], -1), axis=1)
mean_l2 = np.mean(norms)
median_l2 = np.median(norms)

print('Attack results - ASR: {:.3%} - Mean L2: {:.4f} - Median L2: {:.4f}'.format(success_rate, mean_l2, median_l2))
torch.save({'images':ori_image, 'labels':ori_label, 'adv':adv_image, 'adv_preds':adv_preds},
        'mnist_bp-100_attack.pkl')
