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
import src.utils.logging as logging
logger = logging.get_logger(__name__)
import pandas as pd
from torch.utils.data import Dataset
import ipdb
import torch 
from torchvision import transforms
class TinyImageNetTripletDataset(Dataset):
    """Image Base dataset to inherit from"""
    
    def __init__(self, mode, data_root, triplets_dir, datalst_pth = None, triplets_file = None, transform = None):
        self.mode = mode
        assert osp.isdir(data_root), 'Not found: {}'.format(data_root)
        self.data_root = data_root
        assert osp.isdir(triplets_dir), 'Not found: {}'.format(triplets_dir)
        self.triplets_dir = triplets_dir
        self.resampling   = False
        if self.mode == "train":
            with open(osp.join(triplets_dir, "img_lst.txt")) as fi:
                self.imgs_path = [i.strip() for i in fi.readlines()]
            
            with open(osp.join(triplets_dir, "img_cls.txt")) as fi:
                self.imgs_cls  = [int(i.strip()) for i in fi.readlines()]
            
            if (triplets_file == None):
                self.cls_imgs_ids = np.load(osp.join(triplets_dir, "class_imgs_ids.npy"))
                self.resampling   = True
                self.resampling_triplet()
            else:    
                self.triplet_pairs = np.load(osp.join(triplets_dir, triplets_file))

        elif self.mode == "val":
            with open(osp.join(triplets_dir, "val_lst.txt")) as fi:
                self.imgs_path = [osp.join("val", "images", i.strip()) for i in fi.readlines()]
            
            with open(osp.join(triplets_dir, "val_cls.txt")) as fi:
                self.imgs_cls  = [int(i.strip()) for i in fi.readlines()]
        else:
            print("Mode ", self.mode, "does not support")
            raise
        self.transform = transform

    def resampling_triplet(self):
        if (self.resampling == False):
            return
        logger.info("Resampling triplet pairs...")
        cls_imgs_ids = self.cls_imgs_ids
        n_total_cls, n_total_imgs  = cls_imgs_ids.shape 
        simp_triplet_pairs = []
        for pos_cls in range(n_total_cls):
            neg_cls = np.array([i for i in range(n_total_cls) if i != pos_cls])
            pos_imgs = cls_imgs_ids[pos_cls]
            for neg_cls_idx in range(len(neg_cls)):
                neg_imgs = cls_imgs_ids[neg_cls[neg_cls_idx]]
                a, p = np.random.choice(pos_imgs, 2)
                n,   = np.random.choice(neg_imgs, 1)
                simp_triplet_pairs.append([a, p, n])
        simp_triplet_pairs = np.array(simp_triplet_pairs)
        self.triplet_pairs = simp_triplet_pairs
    
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
        "val", "./data/tiny-imagenet-200", "triplet_pairs",
        transform = transforms.ToTensor())
    data, lbl = next(iter(loader))
    print(data.shape, lbl)
