import os
import argparse
import tqdm

import torch
import torch.nn.functional as F
from torch.utils import data
from torch.optim import SGD, lr_scheduler
from torch.backends import cudnn

from torchvision import transforms
from torchvision.datasets import MNIST

from small_cnn import SmallCNN
from utils import AverageMeter, save_checkpoint, requires_grad_

parser = argparse.ArgumentParser(description='MNIST Training')

parser.add_argument('--data', default='data/mnist', help='path to dataset')
parser.add_argument('--workers', default=2, type=int, help='number of data loading workers')
parser.add_argument('--save-folder', '--sf', default='model', help='folder to save state dicts')
parser.add_argument('--save-name', '--sn', default='mnist', help='name for saving the final state dict')
parser.add_argument('--save-freq', '--sfr', type=int, help='save frequency')

parser.add_argument('--batch-size', '-b', default=128, type=int, help='mini-batch size')
parser.add_argument('--epochs', '-e', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--lr-decay', '--lrd', default=0.1, type=float, help='decay for learning rate')
parser.add_argument('--lr-step', '--lrs', type=int, help='step size for learning rate decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float, help='weight decay')
parser.add_argument('--drop', default=0.5, type=float, help='dropout rate of the classifier')

args = parser.parse_args()
print(args)
if args.lr_step is None: args.lr_step = args.epochs

if not os.path.exists(args.save_folder) and args.save_folder:
    os.makedirs(args.save_folder)

DEVICE = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')

transform = transforms.Compose([transforms.ToTensor()])

train_data = MNIST(args.data, train=True, transform=transform, download=True)
train_set = data.Subset(train_data, list(range(55000)))
val_set = data.Subset(train_data, list(range(55000, 60000)))
test_set = MNIST(args.data, train=False, transform=transform, download=True)

train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                               drop_last=True, pin_memory=True)
val_loader = data.DataLoader(val_set, batch_size=100, shuffle=True, num_workers=args.workers, pin_memory=True)
test_loader = data.DataLoader(test_set, batch_size=100, shuffle=True, num_workers=args.workers, pin_memory=True)

model = SmallCNN(drop=args.drop).to(DEVICE)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_decay)

best_acc = 0
best_epoch = 0

for epoch in range(args.epochs):
    cudnn.benchmark = True
    model.train()
    requires_grad_(model, True)
    accs = AverageMeter()
    losses = AverageMeter()
    attack_norms = AverageMeter()

    scheduler.step()
    length = len(train_loader)
    for i, (images, labels) in enumerate(tqdm.tqdm(train_loader, ncols=80)):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        logits = model(images)
        accs.append((logits.argmax(1) == labels).float().mean().item())

        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print('Epoch {} | Training | Loss: {:.4f}, Accs: {:.4f}'.format(epoch, losses.avg, accs.avg))

    cudnn.benchmark = False
    model.eval()
    requires_grad_(model, False)
    val_accs = AverageMeter()
    val_losses = AverageMeter()

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm.tqdm(val_loader, ncols=80)):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            val_accs.append((logits.argmax(1) == labels).float().mean().item())
            val_losses.append(loss.item())

    print('Epoch {} | Validation | Loss: {:.4f}, Accs: {:.4f}'.format(epoch, val_losses.avg, val_accs.avg))

    if val_accs.avg >= best_acc:
        best_acc = val_accs.avg
        best_epoch = epoch
        best_dict = model.state_dict()

    if args.save_freq and not (epoch + 1) % args.save_freq:
        save_checkpoint(
            model.state_dict(), os.path.join(args.save_folder, args.save_name + '_{}.pth'.format(epoch + 1)), cpu=True)

model.load_state_dict(best_dict)

test_accs = AverageMeter()
test_losses = AverageMeter()

with torch.no_grad():
    for i, (images, labels) in enumerate(tqdm.tqdm(test_loader, ncols=80)):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        test_accs.append((logits.argmax(1) == labels).float().mean().item())
        test_losses.append(loss.item())

print('\nTest accuracy with model from epoch {}: {:.4f} with loss: {:.4f}'.format(best_epoch, test_accs.avg,
                                                                                      test_losses.avg))

print('\nSaving model...')
save_checkpoint(model.state_dict(), os.path.join(args.save_folder, args.save_name + '.pth'), cpu=True)
