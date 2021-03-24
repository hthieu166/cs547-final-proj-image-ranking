"""Image Base dataset"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
import os.path as osp
from PIL import Image

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import ipdb
import glob

class ImageBaseDataset(Dataset):
    """Image Base dataset to inherit from"""

    def __init__(self, mode, data_root, datalst_pth = None, lbl_fname = None, lbl_type = None,
                transform = None, spldata_dir = None):
        """Initialize the dataset

        Args:
            mode: `train`, `val`, or `test` mode
            datalst_pth: a dictionary of paths wrt to different modes
            data_root: root directory of the dataset
            lbl_fname: path to the label file
            lbl_type: type of the label to retrieve
            transform: transform object to apply random data augmentation
        """
        self.mode = mode
        self.data_root = data_root
        assert osp.isdir(data_root), 'Not found: {}'.format(data_root)
        if datalst_pth is None:
            self.datalst_pth = None
        else:
            self.datalst_pth = datalst_pth[mode]
            assert osp.isfile(self.datalst_pth), 'Not found: {}'.format(self.datalst_pth)
        
        self.build_dataset()
        # Set up transforms object (#TODO: check again when transforms are necessary)
        self.transforms = transform
    
    def build_dataset(self):
        raise NotImplementedError

    def get_data_sample(self, idx):
        raise NotImplementedError
    
    def get_data_label(self, idx):
        """This function returns a label for image at index idx"""
        raise NotImplementedError

    def get_img_list(self):
        """This function returns image at index idx"""
        return open(self.datalst_pth, 'r').read().splitlines()

    def get_nclasses(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.img_lst)

    def __getitem__(self, idx):
        """Get item wrt a given index
        Args:
            idx: sample index
        Returns:
            imgs: (Image Array)
            lbl: (float) label
        """
        # Retrieve measurement ID wrt to the given index
        lbl = self.get_data_label(idx)
        # Read data and labels coreponding to the measurement ID
        data = self.get_data_sample(idx)
        
        # If transform is not none, apply data augmentation strategy
        if self.transforms:
            data = self.transforms(data)
        return data, lbl
