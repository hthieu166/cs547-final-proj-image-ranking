"""AICity20 VehicleType dataset"""
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
from src.datasets.img_base_dataset import ImageBaseDataset
import ipdb

class TinyImageNetDataset(ImageBaseDataset):
    """Tiny Imagenet Dataset"""
    def get_support_modes(self):
        """This function return support modes for the current dataset
        """
        return ['train', 'test', 'val']

    def build_class_dict(self, cls_file = "wnids.txt"):
        class_name_dict = {}
        with open(osp.join(self.data_root, cls_file), "r") as fi:
            cls_names = fi.readlines()
            for i in range(len(cls_names)):
                class_name_dict[cls_names[i].strip()] = i 
        return class_name_dict

    def build_dataset(self):
        all_imgs = glob.glob(osp.join(self.data_root, self.mode, "*", "images", "*.JPEG"))
        def get_image_name_from_path(img_path):
            return osp.basename(img_path).split('.')[0]
        
        self.class_name_dict = self.build_class_dict()
        def get_label(img_name):
            return self.class_name_dict[img_name.split('_')[0]]

        self.label_dict      = {}
        self.img_path_dict   = {}
        self.img_lst         = []

        for img_path in all_imgs:
            img_name = get_image_name_from_path(img_path)
            self.label_dict[img_name]   = get_label(img_name)
            self.img_path_dict[img_name]= img_path
            self.img_lst.append(img_name)

    def get_data_label(self, idx):
        """This function returns a label for image at index idx"""
        img_id = self.img_lst[idx]
        return self.label_dict[img_id]

    def get_data_sample(self, idx):
        """This function returns image at index idx"""
        img_name = osp.join(self.img_path_dict[self.img_lst[idx]])
        image = Image.open(img_name)
        if (len(image.split()) == 1):
            image = image.convert('RGB')
        return image

    def get_nclasses(self):
        """This function returns the number of vehicle instances"""
        return len(self.ist2idx)

    def get_inst2imgs_dict(self):
        """This function returns a dictionary with the following structure:
            inst2imgs[<instance ID>] = list([img_idx1, img_idx2, ...])
        """
        self.inst2imgs = {}
        for idx in range(len(self.img_fname_lst)):
            lbl = self.get_data_label(idx)
            if lbl not in self.inst2imgs:
                self.inst2imgs[lbl] = []
            self.inst2imgs[lbl].append(idx)
        return self.inst2imgs