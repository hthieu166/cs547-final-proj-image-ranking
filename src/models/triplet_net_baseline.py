from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

"""---- Code By Hieu Hoang ----"""
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from torch import nn
#immport backbone models
import torchvision.models as models 

from src.models.base_model import BaseModel
from src.models.efficient_net.model import EfficientNet
from src.models.my_resnet.model import resnet50

import ipdb
import torch
class TripletNetBaseline(BaseModel):

    def __init__(self, device, base, feat_vect_size = None, pretrained = True):
        """Initialize the model

        Args:
            device: the device (cpu/gpu) to place the tensors
            base: (str) name of base model
            n_classes: (int) number of classes
        """
        super().__init__(device)
        self.base = base
        self.pretrained = pretrained
        self.feat_vect_size = feat_vect_size

        self.build_model()

        
    def set_parameter_requires_grad(self, is_features_extracter):
        """Set requires grad for all model's parameters
        Args:
            is_feature_extracter: (bool) if we only use base model as feature extractor (no training)
        """
        if is_features_extracter:
            for param in self.model.parameters():
                param.requires_grad = False

    def build_model(self):
        """Build model architecture
        """
        base_out_channels = 2048
        # Build backbone
        if (self.base == 'resnet101'):
            resnet = models.resnet101(pretrained = self.pretrained)
            self.model = nn.Sequential(*list(resnet.children())[:-1])
        
        elif (self.base == 'resnet50'):
            self.model = models.resnet50(pretrained = self.pretrained)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        elif (self.base == 'my_resnet50'):
            self.model = resnet50(3, 3)
        
        elif (self.base == 'densenet121'):
            self.model = torch.hub.load('pytorch/vision:v0.9.0', 'densenet121', pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1],
                                        nn.AdaptiveAvgPool2d(1))
            base_out_channels = 1024
        elif ('efficientnet' in self.base):
            self.model = EfficientNet.from_pretrained(self.base, include_top = False)
            ver = self.base.split('-')[1]
            if ver in['b0', 'b1']:
                base_out_channels = 1280
            elif ver in ['b2']:
                base_out_channels = 1408
        else:
            raise ValueError("Model {} is not supported! ".format(self.base))
        # Build fc
        if (self.feat_vect_size != None):
            self.fc = nn.Linear(base_out_channels, self.feat_vect_size)
        else:
            self.fc = None

    def forward(self, input_tensor):
        """Forward function of the model
        Args:
            input_tensor: pytorch input tensor
        """
        out = self.model(input_tensor).squeeze()
        
        if (self.fc != None):
            out = self.fc(out)
        return out
