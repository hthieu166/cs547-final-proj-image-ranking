"""Factory pattern for different models and datasets"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
from torchvision import transforms
from torch import nn
import ipdb
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'src')))

""" > Import your datasets here """
from src.datasets.tiny_imagenet import TinyImageNetTripletDataset
from src.datasets.tiny_imagenet import TinyImageNetDataset

""" > Import your models here """
from src.models.triplet_net import TripletNet
from src.models.triplet_net_baseline import TripletNetBaseline
""" > Import your data augmentation functions here """
from src.data_augmentation.image_random_erasing import RandomErasing

""" > Import your loss functions here """
from src.losses.triplet_loss_baseline import TripletLossBaseline

from src.losses.triplet_loss_batch_hard import TripletLossBatchHard


""" > Import your data samplers here """
from src.samplers.batch_sampler import BatchSampler

import src.utils.logging as logging
logger = logging.get_logger(__name__)

class BaseFactory():
    """Base factory for dataset and model generator"""
    def __init__(self):
        self.info_msg = 'Generating object'
        self.objfn_dict = None

    def get_object(self, name, **kwargs):
        """Generate registered object based on given name and variables"""
        gen_obj = self.objfn_dict[name](**kwargs)
        return gen_obj
        
    def print_check_info(self, name, **kwargs):
        assert name in self.objfn_dict, \
                '{} not recognized. ' \
                'Only support:\n{}'.format(name, self.objfn_dict.keys())
        logger.info('%s: %s' % (self.info_msg, name))
        logger.info('Given parameters:')
        for key, val in kwargs.items():
            logger.info('    %s = %s' % (key, val))
        logger.info('-'*80)
        
    def generate(self, name, **kwargs):
        """Generate object based on the given name and variables
        Args:
            name: a string to describe the type of the object
            kwargs: keyworded variables of the object to generate

        Return:
            Generated object with corresponding type and arguments
        """
        self.print_check_info(name, **kwargs)
        return self.get_object(name, **kwargs)


class ModelFactory(BaseFactory):
    """Factory for model generator"""
    def __init__(self):
        self.info_msg = 'Generating model'
        self.objfn_dict = {
            'TripletNetBaseline': TripletNetBaseline,
        }

class DatasetFactory(BaseFactory):
    """Factory for dataset generator"""
    def __init__(self):
        self.info_msg = 'Generating dataset'
        self.objfn_dict = {
            'TinyImageNetTriplet': TinyImageNetTripletDataset,
            'TinyImageNet': TinyImageNetDataset
        }

class DataAugmentationFactory(BaseFactory):
    """Factory for data augmentation object generator"""
    def __init__(self):
        self.info_msg = 'Generating data augmentation strategy'
        self.objfn_dict = {
            "resize": transforms.Resize,
            "center_crop": transforms.CenterCrop,
            "random_crop": transforms.RandomCrop,
            "to_tensor": transforms.ToTensor,
            "normalize": transforms.Normalize,
            "horizontal_flip": transforms.RandomHorizontalFlip,
            "vertical_flip": transforms.RandomVerticalFlip,
            "random_erasing": RandomErasing,
        }

    def generate(self, data_augment_name, kwargs):
        """Generate data augmentation strategies based on given name and variables"""
        if kwargs is not None:
            self.print_check_info(data_augment_name, **kwargs)
            gen_data_augment = self.objfn_dict[data_augment_name](**kwargs)
        else:
            self.print_check_info(data_augment_name)
            gen_data_augment = self.objfn_dict[data_augment_name]()
        return gen_data_augment


class LossFactory(BaseFactory):
    """Factory for loss function generator"""
    def __init__(self):
        self.info_msg = 'Generating loss function'
        self.objfn_dict = {
            "TripletLossBatchHard": TripletLossBatchHard,
            "TripletLossBaseline": TripletLossBaseline
        }


class DataSamplerFactory(BaseFactory):
    """Factory for loss function generator"""
    def __init__(self):
        self.info_msg = 'Generating data sampler'
        self.objfn_dict = {
            "BatchSampler": BatchSampler,
        }