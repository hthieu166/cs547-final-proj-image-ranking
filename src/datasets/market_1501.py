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
        assert osp.isdir(data_root), 'Not found: {}'.format(data_root)

        if self.mode == "train":
            self.data_root = osp.join(data_root, "bounding_box_train")
        elif self.mode == "test":
            self.data_root = osp.join(data_root, "bounding_box_test")
        elif self.mode == "query":
            self.data_root = osp.join(data_root, "query")
        else:
            raise "Mode does not support" 
        self.transform = transform
        self.name = "Market1501Dataset"
        self.imgs_path = glob.glob(osp.join(self.data_root, "*.jpg"))
        self.imgs_cls  = [int(osp.basename(img_path).split('_')[0]) for img_path in self.imgs_path]

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
        return np.unique(self.imgs_cls)

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
    dataset = Market1501(
        "train", "/home/hthieu/data/Market-1501-v15.09.15/",
        transform = transforms.ToTensor())
    print("Train #id:", dataset.get_nclasses())
    dataset = Market1501(
        "test", "/home/hthieu/data/Market-1501-v15.09.15/",
        transform = transforms.ToTensor())
    print("Test #id:", dataset.get_nclasses())
    dataset = Market1501(
        "query", "/home/hthieu/data/Market-1501-v15.09.15/",
        transform = transforms.ToTensor())
    print("Query #id:", dataset.get_nclasses())
    data, lbl = next(iter(dataset))
    print(data.shape, lbl)