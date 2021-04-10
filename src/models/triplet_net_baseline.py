"""Dummy model"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from torch import nn
#immport base model here:
import torchvision.models as models 

from src.models.base_model import BaseModel
import ipdb

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
        # Build backbone
        if (self.base == 'resnet101'):
            resnet = models.resnet101(pretrained = self.pretrained)
            self.model = nn.Sequential(*list(resnet.children())[:-1])
        elif (self.base == 'densenet161'):
            self.model = models.resnet101(pretrained = self.pretrained)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        elif (self.base == 'resnet50'):
            self.model = models.resnet50(pretrained = self.pretrained)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        else:
            raise ValueError("Model {} is not supported! ".format(self.base))
        # Build fc
        if (self.feat_vect_size != None):
            self.fc = nn.Linear(2048, self.feat_vect_size)
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
