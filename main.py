"""Main code for train/val/test"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
import argparse

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from src.utils.load_cfg import ConfigLoader
from src.factories import ModelFactory, LossFactory, DataSamplerFactory
from src.loaders.base_loader_factory import BaseDataLoaderFactory
from trainer import train
from tester import *
import src.utils.logging as logging
logger = logging.get_logger(__name__)
import ipdb
import src.config as cfg
import os.path as osp

def parse_args():
    """Parse input arguments"""
    def str2bool(v):
        """Convert a string to boolean type"""
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset_cfg', type=str,
        help='Path to the dataset config filename')
    parser.add_argument(
        '-t', '--train_cfg', type=str,
        help='Path to the training config filename')

    parser.add_argument(
        '-i', '--is_training', type=str2bool,
        help='Whether is in training or testing mode')
    parser.add_argument(
        '-m', '--train_mode', type=str,
        choices=['from_scratch', 'from_pretrained', 'resume'],
        help='Which mode to start the training from')
    parser.add_argument(
        '-l', '--logdir', type=str,
        help='Directory to store the log')
    parser.add_argument(
        '--log_fname', type=str, default=None,
        help='Path to the file to store running log (beside showing to stdout)')

    parser.add_argument(
        '-w', '--num_workers', type=int, default=4,
        help='Number of threads for data loading')
    parser.add_argument(
        '-g', '--gpu_id', type=int, default=-1,
        help='Which GPU to run the code')

    parser.add_argument(
        '--pretrained_model_path', type=str, default='',
        help='Path to the model to test. Only needed if is not training or '
             'is training and mode==from_pretrained')

    args = parser.parse_args()

    if (not args.is_training) or \
            (args.is_training and args.train_mode == 'from_pretrained'):
        assert os.path.isfile(args.pretrained_model_path), \
            'pretrained_model_path not found: {}'.format(args.pretrained_model_path)
    return args


def main():
    """Main function"""
    # Load configurations
    model_name, model_params = ConfigLoader.load_model_cfg(args.train_cfg)
    dataset_name, dataset_params = ConfigLoader.load_dataset_cfg(args.dataset_cfg)
    train_params = ConfigLoader.load_train_cfg(args.train_cfg)
    
    # Set up device (cpu or gpu)
    if args.gpu_id < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.gpu_id)
    cfg.device = device
    logger.info('Using device: %s' % device)

    # Build model
    model_factory = ModelFactory()
    model = model_factory.generate(model_name, device=device, **model_params)
    model = model.to(device)
    
    # Setup loss criterion
    loss_fn_factory = LossFactory()
    loss_n = list(train_params['loss_fn'].keys())[0]
    loss_params = list(train_params['loss_fn'].values())[0]
    criterion = loss_fn_factory.generate(loss_n, **loss_params)

    # Setup common parameters for data loaders (shared among train/val/test)
    common_loader_params = {
        'batch_size': train_params['batch_size'],
        'num_workers': args.num_workers,
    }
    # Setup train samplers
    train_sampler_name   = None
    train_sampler_params = None
    if 'sampler' in train_params:
        train_sampler_name = list(train_params['sampler'].keys())[0]
        train_sampler_params = list(train_params['sampler'].values())[0]

    # Setup loaders
    loader_fact = BaseDataLoaderFactory(dataset_name, dataset_params, train_params, common_loader_params)
    # Main pipeline
    if args.is_training:
        train_val_loaders = {
            "train": loader_fact.build_loader("train", sampler_name = train_sampler_name,
                                                      sampler_params= train_sampler_params)
        }
        if  dataset_name in ["TinyImageNetTriplet", "TinyImageNet"]:
            train_val_loaders["val"] = loader_fact.build_loader("val",  do_shuffle= False, do_drop_last = False)
        elif dataset_name in ["Market1501"]:
            train_val_loaders["que"] = loader_fact.build_loader("que",  do_shuffle= False, do_drop_last = False)
            train_val_loaders["gal"] = loader_fact.build_loader("test", do_shuffle= False, do_drop_last = False)
        
        # Create optimizer
        if train_params["optimizer_name"] == "Adam":
            logger.info("Using Adam optimizer")
            optimizer = optim.Adam(model.parameters(),
                lr=train_params['init_lr'], **train_params["optimizer_params"], 
                weight_decay = train_params["weight_decay"])
        elif train_params["optimizer_name"] == "SGD":
            logger.info("Using SGD optimizer")
            optimizer = optim.SGD(model.parameters(),
                lr=train_params['init_lr'], **train_params["optimizer_params"], 
                weight_decay = train_params["weight_decay"])
        else:
            print("Invalid optimizer request")
            raise
        # Train/val routine
        train(model, optimizer, criterion, train_val_loaders, args.logdir,
              args.train_mode, train_params, device, args.pretrained_model_path)
    else:
        model.load_model(args.pretrained_model_path)
       
        if  dataset_name in ["TinyImageNetTriplet", "TinyImageNet"]:
             # Create data loader for testing
            test_loaders = loader_fact.build_loader("val",  do_shuffle= False, do_drop_last = False)
            eval_res = test(model, criterion, test_loaders, device, export_result = True)
        elif dataset_name in ["Market1501"]:
            que_loader = loader_fact.build_loader("que",  do_shuffle= False, do_drop_last = False)
            gal_loader = loader_fact.build_loader("test", do_shuffle= False, do_drop_last = False)
            eval_res = test_que_gal(model, criterion, 
                    que_loader, gal_loader, device, export_result = True)
       
        eval_dir = osp.join(args.logdir, "eval_results")
        os.makedirs(eval_dir, exist_ok = True)
        np.save(osp.join(eval_dir, "val_preds.npy"), eval_res["preds"])   
    return 0

if __name__ == '__main__':
    # Fix random seeds here for pytorch and numpy
    torch.manual_seed(1)
    np.random.seed(2)

    # Parse input arguments
    args = parse_args()

    # Setup logging format
    logging.setup_logging(args.log_fname)

    # Run the main function
    sys.exit(main())