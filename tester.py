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
    with torch.no_grad():
        for i, (samples, labels) in enumerate(loader):
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
    return img_ranking_evaluate(test_img_embs, test_img_lbls, export_result)

if __name__ == "__main__":
    pass