"""Validation/Testing routine"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch

from src.utils.misc import MiscUtils
import src.utils.logging as logging
from src.utils.img_ranking_evaluate import *
logger = logging.get_logger(__name__)

import ipdb

def test(model, criterion, loader, device, export_result = False):
    """Evaluate the performance of a model

    Args:
        model: model to evaluate
        criterion: loss function
        loader: dictionary of data loaders for testing
        device: id of the device for torch to allocate objects
    Return:
        test_loss: average loss over the test dataset
        test_score: score over the test dataset
    """
    # Switch to eval mode
    model.eval()
    # Prepare data loader
    assert loader is not None, "Evaluation loader is not specified"
    # Setup progressbar
    pbar = MiscUtils.gen_pbar(max_value=len(loader), msg="Testing: ")
    test_img_embs = []
    test_img_lbls = []
    # ipdb.set_trace()
    with torch.no_grad():
        for i, (samples, labels) in enumerate(loader):
            # ipdb.set_trace()
            # Evaluating for the current batch
            samples = samples.to(device)
            labels = labels.to(device)
            # Forwarding
            outputs = model(samples)
            test_img_embs.append(outputs)
            test_img_lbls.append(labels)
            # Monitor progress
            pbar.update(i+1)

    test_img_embs = torch.vstack(test_img_embs)
    test_img_lbls = torch.hstack(test_img_lbls)
    pbar.finish()
    return img_ranking_evaluate_tinyImageNet(test_img_embs, test_img_lbls, export_result)

def test_que_gal(model, criterion, que_loader, gal_loader, device, export_result = False):
    """Evaluate the performance of a model with test and query dataset

    Args:
        model: model to evaluate
        criterion: loss function
        loader: dictionary of data loaders for testing
        device: id of the device for torch to allocate objects
    Return:
        test_loss: average loss over the test dataset
        test_score: score over the test dataset
    """
    # Switch to eval mode
    model.eval()
    # Prepare data loader
    assert gal_loader is not None, "Evaluation loader is not specified"
    assert que_loader  is not None, "Query loader is not specified"
    # Setup progressbar
    pbar = MiscUtils.gen_pbar(max_value=len(test_loader) + len(gal_loader), msg="Testing: ")
    que_img_embs = []
    que_img_lbls = []
    gal_img_embs = []
    gal_img_lbls = []
    
    with torch.no_grad():
        # Embs gallery images
        for i, (samples, labels) in enumerate(gal_loader):
            samples = samples.to(device)
            labels = labels.to(device)
            outputs = model(samples)
            gal_img_embs.append(outputs)
            gal_img_lbls.append(labels)
            # Monitor progress
            pbar.update(i+1)
        
        # Embs query images
        for i, (samples, labels) in enumerate(que_loader):
            samples = samples.to(device)
            labels = labels.to(device)
            outputs = model(samples)
            que_img_embs.append(outputs)
            que_img_lbls.append(labels)
            # Monitor progress
            pbar.update(len(gal_loader) + i + 1)

    gal_img_embs = torch.vstack(gal_img_embs)
    gal_img_lbls = torch.hstack(gal_img_lbls)
    que_img_embs = torch.vstack(que_img_embs)
    que_img_lbls = torch.hstack(que_img_lbls)
    pbar.finish()
    return img_ranking_evaluate_Market1501(
        gal_img_embs, gal_img_lbls, que_img_embs, que_img_lbls, export_result)

if __name__ == "__main__":
    pass