# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 15:57:01 2017

@author: bbrattol
"""
import torch
import torch.nn as nn
from torch import cat

import sys
sys.path.append('../Utils')
from Layers import LRN

class Network(nn.Module):

    def __init__(self, num_classes=21, groups = 2):
        super(Network, self).__init__()
        
        self.conv = nn.Sequential()
        self.conv.add_module('conv1_s1',nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0))
        self.conv.add_module('relu1_s1',nn.ReLU(inplace=True))
        #self.conv.add_module('bn1_s1',nn.BatchNorm2d(96))
        self.conv.add_module('pool1_s1',nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv.add_module('lrn1_s1',LRN(local_size=5, alpha=0.0001, beta=0.75))
        
        self.conv.add_module('conv2_s1',nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=groups))
        self.conv.add_module('relu2_s1',nn.ReLU(inplace=True))
        #self.conv.add_module('bn2_s1',nn.BatchNorm2d(256))
        self.conv.add_module('pool2_s1',nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv.add_module('lrn2_s1',LRN(local_size=5, alpha=0.0001, beta=0.75))
        
        self.conv.add_module('conv3_s1',nn.Conv2d(256, 384, kernel_size=3, padding=1))
        self.conv.add_module('relu3_s1',nn.ReLU(inplace=True))
        #self.conv.add_module('bn3_s1',nn.BatchNorm2d(384))
        
        self.conv.add_module('conv4_s1',nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=groups))
        #self.conv.add_module('bn4_s1',nn.BatchNorm2d(384))
        self.conv.add_module('relu4_s1',nn.ReLU(inplace=True))
        
        self.conv.add_module('conv5_s1',nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=groups))
        #self.conv.add_module('bn5_s1',nn.BatchNorm2d(256))
        self.conv.add_module('relu5_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool5_s1',nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1',nn.Linear(256*6*6, 4096))
        self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
        self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))
        
        self.fc7 = nn.Sequential()
        self.fc7.add_module('fc7',nn.Linear(4096,4096))
        self.fc7.add_module('relu7',nn.ReLU(inplace=True))
        self.fc7.add_module('drop7',nn.Dropout(p=0.5))
        
        self.classifier = nn.Sequential()
        self.classifier.add_module('fc8',nn.Linear(4096, num_classes))
    
    def load(self,checkpoint,load_fc=False):
        model_dict = self.state_dict()
        layers = [k for k, v in model_dict.items()]
        
        pretrained_dict = torch.load(checkpoint)
        keys = [k for k, v in pretrained_dict.items()]
        keys.sort()
        #keys = keys[2:-4] #load until conv5
        
        to_load = []
        for k in keys:
            if k not in model_dict:
                continue
#            if 'conv5' in k or 'bn5' in k:
#                continue
            if 'conv' in k:
                to_load.append(k)
            if 'fc' in k and load_fc:
                to_load.append(k)
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in to_load and k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
    
    def save(self,checkpointFold,epoch):
        filename = '%s/jps_%03i.pth.tar'%(checkpointFold,epoch)
        torch.save(self.state_dict(), filename)
    
    def forward(self, x):
        B,C,H,W = x.size()
        x = self.conv(x)
        x = self.fc6(x.view(B,-1))
        x = self.fc7(x)
        x = self.classifier(x)
        return x
    
