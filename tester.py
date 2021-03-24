"""Validation/Testing routine"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch

from src.utils.misc import MiscUtils
from src.inferences.base_infer import BaseInfer
import src.utils.logging as logging
logger = logging.get_logger(__name__)
import ipdb

def test(model, criterion, loaders, device):
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
    eval_loader = loaders['test'] if 'test' in loaders else None
    assert eval_loader is not None, "Evaluation loader is not specified"
    
    test_loss = 0.0
    # Setup progressbar
    pbar = MiscUtils.gen_pbar(max_value=len(eval_loader), msg=eval_mess)
    with torch.no_grad():
        for i, (samples, labels) in enumerate(eval_loader):
            # Evaluating for the current batch
            samples = samples.to(device)
            labels = labels.to(device)
            # Forwarding
            outputs = model(samples)
            loss = 0.0 #self.criterion(outputs, labels)
            test_loss += loss
            # Monitor progress
            pbar.update(i+1)
    pbar.finish()
    
