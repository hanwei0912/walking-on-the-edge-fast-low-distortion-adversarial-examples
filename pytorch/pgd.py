import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb

class PGD:
    def __init__(self,
                steps,
                eps,
                eps_iter,
                device=torch.device('cpu')):
        self.steps = steps
        self.device = device
        self.eps = eps
        self.eps_iter=eps_iter

    def attack(self, model, inputs, labels, targeted=False):
        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')
        batch_size = inputs.shape[0]
        multiplier = 1 if targeted else -1
        #adv = torch.ones_like(inputs, requires_grad=True)*inputs
        delta = torch.zeros_like(inputs, requires_grad=True)

        for i in range(self.steps):
            adv = delta + inputs
            logits = model(adv)
            pred_labels = logits.argmax(1)
            #ce_loss = F.nll_loss(logits, labels)
            ce_loss = F.cross_entropy(logits, labels,reduction='sum')
            loss = multiplier * ce_loss
            delta.retain_grad()
            model.zero_grad()
            #loss.backward()
            loss.backward(retain_graph=True)
            gra = delta.grad
            grad_norm = gra.view(batch_size, -1).norm(p=2, dim=1)
            grad_norms = grad_norm.view(batch_size,1,1,1).expand(-1,gra.shape[1],gra.shape[2],gra.shape[3])
            ng = gra/grad_norms
            delta = ng*self.eps_iter

            adv = adv + delta
            adv = torch.clamp(adv,0,1)
            diff = adv - inputs
            #diff = diff.to(self.device)
            diff_norm = diff.view(batch_size,-1).norm(p=2,dim=1)
            diff_norm = diff_norm.view(batch_size,1,1,1).expand(-1,gra.shape[1],gra.shape[2],gra.shape[3])
            delta = self.eps * (diff/diff_norm)

        return adv





