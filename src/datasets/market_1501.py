from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

"""---- Code By Hieu Hoang ----"""
import sys
import os
import xml.etree.ElementTree as ET
import numpy as np
import os.path as osp
from PIL import Image
import glob
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
# import src.utils.logging as logging
# logger = logging.get_logger(__name__)
import pandas as pd
from torch.utils.data import Dataset
import ipdb
import torch 
from torchvision import transforms
class Market1501(Dataset):
    def __init__(self, mode, data_root, datalst_pth = None, transform = None):
        self.mode = mode
        self.val_mode = False
        assert osp.isdir(data_root), 'Not found: {}'.format(data_root)
        if self.mode == "trainval":
            self.data_root = osp.join(data_root, "bounding_box_train")
        elif self.mode == "test":
            self.data_root = osp.join(data_root, "bounding_box_test")
        elif self.mode == "que":
            self.data_root = osp.join(data_root, "query")
        elif self.mode == "train" or  self.mode == "val":
            self.data_root = osp.join(data_root, "bounding_box_train")
            self.val_mode  = True
        else:
            raise "Mode does not support" 
        self.transform = transform
        self.name = "Market1501Dataset"
        self.imgs_path = np.array(glob.glob(osp.join(self.data_root, "*.jpg")))
        self.imgs_cls  = np.array([int(osp.basename(img_path).split('_')[0]) for img_path in self.imgs_path])

        if self.val_mode == True:
            uniqe_lbls = self.get_unique_labels()
            val_lbls    = uniqe_lbls[::7]
            val_idx     = np.isin(self.imgs_cls, val_lbls)
            train_idx   = ~val_idx
            slc_idx     = train_idx if self.mode == "train" else val_idx
            self.imgs_path = self.imgs_path[slc_idx]
            self.imgs_cls = self.imgs_cls[slc_idx]

    def resampling_triplet(self):
        pass
    
    def read_image_by_id(self, idx):
        """This function returns image at index idx"""
        img_name = osp.join(self.data_root, self.imgs_path[idx])
        image = Image.open(img_name)
        if (len(image.split()) == 1):
            image = image.convert('RGB')
        return image
    
    def get_list_of_labels(self):
        return self.imgs_cls
    
    def get_nclasses(self):
        return len(self.get_unique_labels())

    def get_unique_labels(self):
        ulbl = np.unique(self.imgs_cls)
        ulbl.sort()
        return ulbl

    def __len__(self):
        return len(self.imgs_cls)
    
    def get_inst2imgs_dict(self):
        inst2img_dict = {k: list(np.where(self.imgs_cls == k )[0]) for k in self.get_unique_labels()}
        return inst2img_dict

    def __getitem__(self, idx):
        """Get item wrt a given index
        Args:
            idx: sample index
        Returns:
            img: (Image Array)
            cls: (float) label
        """
        img = self.read_image_by_id(idx)
        if self.transform:
            img = self.transform(img)
        return img, self.imgs_cls[idx]
       
if __name__ == "__main__":
    print("Market-1501")
    root_dir = "/mnt/data0-nfs/shared-datasets/Market-1501-v15.09.15/"
    dataset = Market1501(
        "trainval",root_dir,
        transform = transforms.ToTensor())
    print("Trainval #id:", dataset.get_nclasses())
    
    dataset = Market1501(
        "test", root_dir,
        transform = transforms.ToTensor())
    print("Test #id:", dataset.get_nclasses())
   
    dataset = Market1501(
        "que", root_dir,
        transform = transforms.ToTensor())
    print("Query #id:", dataset.get_nclasses())
    
    dataset = Market1501(
        "train", root_dir,
        transform = transforms.ToTensor())
    print(len(dataset))
    # print(dataset.get_unique_labels())
    print("Train #id:", dataset.get_nclasses())
    
    dataset = Market1501(
        "val", root_dir,
        transform = transforms.ToTensor())
    print(len(dataset))
    # print(dataset.get_unique_labels())
    print("Val #id:", dataset.get_nclasses())
    data, lbl = next(iter(dataset))
    print(data.shape, lbl)