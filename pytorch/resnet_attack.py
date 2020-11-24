import torch
import numpy as np
from foolbox.models import PyTorchModel as ptm
from torchvision import models, datasets
from torch.utils.data import DataLoader, Dataset, TensorDataset
import cv2
import torch.nn as nn
from foolbox.criteria import TargetClassProbability, TargetClass, Misclassification
from foolbox.attacks import PGD, FGSM, CarliniWagnerL2Attack, L1BasicIterativeAttack, AdamL2BasicIterativeAttack
import foolbox
from foolbox.distances import Linfinity, MeanSquaredDistance, MAE, MSE
import matplotlib.pyplot as plt
import time

class Preprocessing_Layer(torch.nn.Module):
    def __init__(self, mean, std):
        super(Preprocessing_Layer, self).__init__()

        self.mean = mean

        self.std = std

    def preprocess(self, img, mean, staired_adv):
        image = img.clone()
        image /= 255.0
        image = image.transpose(1, 3).transpose(2, 3)
        image[:,0,:,:] = (image[:,0,:,:] - mean[0]) / std[0]
        image[:,1,:,:] = (image[:,1,:,:] - mean[1]) / std[1]
        image[:,2,:,:] = (image[:,2,:,:] - mean[2]) / std[2]

        return(image)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        res = self.preprocess(x, self.mean, self.std)
        return res


model_name = 'resnet18'

#img_paths = ['images/car.jpg']#,'images/car.jpg','images/boat.jpg']#,'images/bird.jpg','images/horse.jpg','images/flower.jpg' ,'images/goldfish.jpg']

 

IMG_SIZE = 224

 

canvas = np.zeros((len(img_paths),224,224,3))

for i in range(len(img_paths)):


    orig = cv2.imread(img_paths[i])[..., ::-1]


    orig = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))


    canvas[i,:,:,:] = orig

mean = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.225]

model_init = getattr(models,model_name)(pretrained=True)

preprocess_layer = Preprocessing_Layer(mean,std)

model = nn.Sequential(preprocess_layer, model_init)

model.eval()


