import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb

class IFGSM:
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
            adv = delta.clone() + inputs
            adv = torch.autograd.Variable(adv, requires_grad=True)
            logits = model(adv)
            pred_labels = logits.argmax(1)
            #loss = F.nll_loss(logits, labels)
            ce_loss = nn.CrossEntropyLoss()
            loss = ce_loss(logits,torch.autograd.Variable(labels))
            loss = multiplier * ce_loss
            #delta.retain_grad()
            #model.zero_grad()
            #loss.backward()
            loss.backward(retain_graph=False)

            delta = self.eps_iter * adv.grad.sign_()
            adv.grad.data.zero_()

            adv = adv + delta
            adv = torch.clamp(adv,0,1)
            diff = adv - inputs
            delta = torch.clamp(diff, -self.eps, self.eps)
            model.zero_grad()
        return adv

