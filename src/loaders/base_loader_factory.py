import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from src.factories import DataAugmentationFactory, DataSamplerFactory, DatasetFactory
import ipdb

class BaseDataLoaderFactory():
    def __init__(self, dataset_name, dataset_params, train_params, base_loader_params):
        # Copy some parameters
        self.dataset_name = dataset_name
        self.dataset_params = dataset_params
        self.train_params = train_params
        self.base_loader_params = base_loader_params
        # Generate factories
        self.data_augment_fact = DataAugmentationFactory()
        self.data_fact = DatasetFactory()
        self.sampler_fact = DataSamplerFactory()
    
        # Init
        self.ld_dict = {'train': {}, 'val':{}, 'test': {}}
        # self.get_data_split()
        self.build_base_parameters()
        self.gen_data_augmentation()
        self.gen_data_sampler()

    def get_data_split(self):
        self.ld_dict['train'] = {}
        self.ld_dict['val']   = {}
        self.ld_dict['test']  = {}

    def set_all(self, key, val):
        self.ld_dict['train'][key] = val
        self.ld_dict['test'][key] = val
        self.ld_dict['val'][key] = val

    def build_base_parameters(self):
        self.set_all("ld_params", dict(self.base_loader_params))
        self.set_all("transform", None)
        self.set_all("sampler", None)
  
    def gen_data_augmentation(self):
        if 'transforms' not in self.train_params:
            return
        else:
            #base transform for all sets
            if ("base" in self.train_params['transforms']):
                self.set_all("transform", 
                    self.train_params['transforms']['base'])
            for targ in self.train_params['transforms']:
                if targ == "base":
                    continue
                else:
                    trsf = self.train_params['transforms'][targ]
                    tm = targ.split('/')
                    self.ld_dict[tm[0]][tm[1]]['transform'] = \
                    {
                        # **self.ld_dict[tm[0]][tm[1]]['transform'],
                        **trsf
                    }
            
    def gen_data_sampler(self):
        if 'samplers' not in self.train_params:
            return
        else:
            for targ in self.train_params['samplers']:
                tm = targ.split('/')
                self.ld_dict[tm[0]][tm[1]]['sampler'] = \
                        self.train_params['samplers'][targ] #train_val/train
    
    # def fix_dataset_params(self,dict_key):
    #     new_dict = {}
    #     data_dict = self.dataset_params[dict_key]
    #     for group in data_dict.keys():
    #         for mem in data_dict[group]:
    #             new_key = group + "/" + mem
    #             new_dict[new_key] = data_dict[group][mem]
    #     self.dataset_params[dict_key] = new_dict

    def get_sampler(self, sampler_config, dataset):
        if (sampler_config == None):
            return None
        sampler_params = list(sampler_config.values())[0]
        sampler_params['dataset'] = dataset
        return self.sampler_fact.generate(
            list(sampler_config.keys())[0],
            **sampler_params
        )
    def get_transform(self, transform_config):
        composed_transforms = transforms.Compose([
        self.data_augment_fact.generate(
            i, transform_config[i])
        for i in transform_config.keys()])
        return composed_transforms

    def build_loader(self, mode, sampler_name = None, sampler_params = None, do_shuffle= True, do_drop_last = True):
        _ld_dict = self.ld_dict[mode]
        #Set shuffle and drop_last
        _shuffle, _drop_last = do_shuffle, do_drop_last
        
        # Add transform
        _transform = self.get_transform(_ld_dict['transform'])
        _data_params = self.dataset_params
        _data_params["transform"] = _transform #add transform to dataset params
        
        # Create dataset
        _dataset = self.data_fact.generate(
            self.dataset_name, mode=mode, **_data_params)
        
        #Use sampler
        _sampler = None
        if sampler_name != None:
            sampler_params["dataset"] = _dataset
            _sampler = self.sampler_fact.generate(sampler_name, **sampler_params)
        
        if (_sampler is not None):
            _loader = DataLoader(_dataset, batch_sampler=_sampler, 
            num_workers= _ld_dict["ld_params"]['num_workers'])
        else:
            _loader = DataLoader(_dataset, shuffle=_shuffle, drop_last=_drop_last, 
            **_ld_dict["ld_params"])
        return _loader

            
        
                

    
        



