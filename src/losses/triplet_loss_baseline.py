
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import torch
import numpy as np
import torch.nn as nn
import ipdb

class TripletLossBaseline(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin = 1.0):
        super(TripletLossBaseline, self).__init__()
        self.margin = margin
        self.dist   = nn.PairwiseDistance(p = 2)

    def forward(self, embeds, labels):
        a_feat = embeds['a_feat']
        p_feat = embeds['p_feat']
        n_feat = embeds['n_feat']
        batch_size = a_feat.shape[0]
        p_dist = self.dist (a_feat, p_feat)
        n_dist = self.dist (a_feat, n_feat)
        loss = self.margin + p_dist - n_dist
        loss[loss<0.0] = 0.0
        loss = loss.mean()
        return loss