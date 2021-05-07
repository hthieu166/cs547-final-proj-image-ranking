''' ---- By Dong Kai Wang ----
    Based on Cheng et al. 
    Person Re-Identification by Multi-Channel Parts-Based CNN
    with Improved Triplet Loss Function
'''

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

class TripletLossImproved(nn.Module):
    def __init__(self, margin_inter = 1.0, margin_intra = 0.01, beta = 0.002):
        super(TripletLossImproved, self).__init__()
        self.margin_inter = margin_inter
        self.margin_intra = margin_intra
        self.beta = beta
        self.dist = nn.PairwiseDistance(p = 2)

    def forward(self, embeds, labels):
        a_feat = embeds['a_feat']
        p_feat = embeds['p_feat']
        n_feat = embeds['n_feat']
        batch_size = a_feat.shape[0]
        p_dist = self.dist (a_feat, p_feat)
        n_dist = self.dist (a_feat, n_feat)
        # Inter-class Loss
        loss_inter = self.margin_inter + p_dist - n_dist
        loss_inter[loss_inter < 0.0] = 0.0
        loss_inter = loss_inter.mean()
        # Intra-class Loss
        loss_intra = p_dist - self.margin_intra
        loss_intra[loss_intra < 0.0] = 0.0
        loss_intra = loss_intra.mean()
        # Combined
        loss = loss_inter + self.beta * loss_intra
        return loss