"""---- By Hieu Hoang ----"""

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
""" Implemented by Hieu Hoang """
class TripletLossBatchAll(nn.Module):
    '''
    Implementation of triplet loss with batch-all sampling
    '''
    def __init__(self, margin = 1.0):
        super(TripletLossBatchAll, self).__init__()
        self.margin = margin

    def forward(self, embeds, labels):
        dist_mtx = torch.cdist(embeds, embeds)
        labels   = labels.unsqueeze(1)
        eq_lbls  = labels == labels.T
        neg_msk  = torch.logical_not(eq_lbls)
        eq_lbls.fill_diagonal_(False)
        pos_msk  = eq_lbls

        pdist = dist_mtx.unsqueeze(2) - dist_mtx.unsqueeze(1) + self.margin 
        triplet_msk = pos_msk.unsqueeze(2) * neg_msk
        triplet_lss = pdist * triplet_msk
        triplet_lss[triplet_lss < 0.0] = 0.0
        #Reduce mean
        non_zero_lss= triplet_lss > 1e-16
        loss = triplet_lss.sum() / (non_zero_lss.sum() + 1e-16)
        return loss

if __name__ == "__main__":
    embeds = torch.randn(9,32)
    labels = torch.Tensor([1,1,1,2,2,2,3,3,3]) 
    lss = TripletLossBatchAll(1.0)
    print(lss(embeds, labels))
