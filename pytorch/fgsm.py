import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pdb

class FGSM:
    def __init__(self,
                eps,
                device=torch.device('cpu')):
        self.device = device
        self.eps = eps

    def attack(self, model, inputs, labels):
        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')
        batch_size = inputs.shape[0]
        delta = torch.zeros_like(inputs, requires_grad=True)
        adv = inputs+delta

        logits = model(adv)
        pred_labels = logits.argmax(1)
        loss = F.nll_loss(logits, labels)
        #loss = F.cross_entropy(logits, labels, reduction='mean')
        model.zero_grad()
        loss.backward()
        grad = delta.grad.sign()
        delta = self.eps*grad

        adv = adv + delta
        adv = torch.clamp(adv,0,1)

        return adv





