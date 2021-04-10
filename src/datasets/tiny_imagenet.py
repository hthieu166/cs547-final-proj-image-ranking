"""TinyImageNet-200 dataset dataset"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
import xml.etree.ElementTree as ET
import numpy as np
import os.path as osp
from PIL import Image
import glob
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import pandas as pd
from torch.utils.data import Dataset
import ipdb
import torch 
from torchvision import transforms
class TinyImageNetTripletDataset(Dataset):
    """Image Base dataset to inherit from"""
    
    def __init__(self, mode, data_root, triplets_dir, datalst_pth = None, lbl_fname = None, lbl_type = None, transform = None):
        self.mode = mode
        assert osp.isdir(data_root), 'Not found: {}'.format(data_root)
        self.data_root = data_root
        assert osp.isdir(triplets_dir), 'Not found: {}'.format(triplets_dir)
        self.triplets_dir = triplets_dir
        if self.mode == "train":
            with open(osp.join(triplets_dir, "img_lst.txt")) as fi:
                self.imgs_path = [i.strip() for i in fi.readlines()]
            
            with open(osp.join(triplets_dir, "img_cls.txt")) as fi:
                self.imgs_cls  = [int(i.strip()) for i in fi.readlines()]

            self.triplet_pairs = np.load(osp.join(triplets_dir, "simp_triplet_samples.npy"))
        elif self.mode == "val":
            with open(osp.join(triplets_dir, "val_lst.txt")) as fi:
                self.imgs_path = [osp.join("val", "images", i.strip()) for i in fi.readlines()]
            
            with open(osp.join(triplets_dir, "val_cls.txt")) as fi:
                self.imgs_cls  = [int(i.strip()) for i in fi.readlines()]
        else:
            print("Mode ", self.mode, "does not support")
            raise
        self.transform = transform

    def read_image_by_id(self, idx):
        """This function returns image at index idx"""
        img_name = osp.join(self.data_root, self.imgs_path[idx])
        image = Image.open(img_name)
        if (len(image.split()) == 1):
            image = image.convert('RGB')
        return image
    
    def get_train_sample(self, idx):
        a_id, p_id, n_id = self.triplet_pairs[idx]
        neg_cls, pos_cls = self.imgs_cls[p_id], self.imgs_cls[n_id]
        data = []
        for im_id in [a_id, p_id, n_id]:
            img = self.read_image_by_id(im_id)
            # If transform is not none, apply data augmentation strategy
            if self.transform:
                img = self.transform(img)
            data.append(img[None,...])
        data = torch.cat(data, dim = 0)
        return data, torch.tensor([pos_cls, neg_cls])
    
    def get_test_sample(self, idx):
        img = self.read_image_by_id(idx)
        if self.transform:
            img = self.transform(img)
        return img, self.imgs_cls[idx]

    def __len__(self):
        if self.mode == "train":
            return len(self.triplet_pairs)
        else:
            return len(self.imgs_path)
    
    def __getitem__(self, idx):
        """Get item wrt a given index
        Args:
            idx: sample index
        Returns:
            (a, p, n): (Image Array)
            (clss_a, clss_p, clss_n): (float) label
        """
        if (self.mode == "train"):
            return self.get_train_sample(idx)
        else:
            return self.get_test_sample(idx)
       
if __name__ == "__main__":
    loader = TinyImageNetTripletDataset(
        "val", "/home/hthieu/data/tiny-imagenet-200", "triplet_pairs",
        transform = transforms.ToTensor())
    data, lbl = next(iter(loader))
    print(data.shape, lbl)
