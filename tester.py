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


def test(model, criterion, gal_loader, device, que_loader = None, export_result = False):
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
    
    # Setup progressbar
    max_val = len(que_loader) + len(gal_loader) if  que_loader != None else len(gal_loader)
    
    pbar = MiscUtils.gen_pbar(max_value = max_val, msg="Testing: ")
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
        gal_img_embs = torch.vstack(gal_img_embs)
        gal_img_lbls = torch.hstack(gal_img_lbls)
        
        if que_loader != None:
            # Embs query images
            for i, (samples, labels) in enumerate(que_loader):
                samples = samples.to(device)
                labels = labels.to(device)
                outputs = model(samples)
                que_img_embs.append(outputs)
                que_img_lbls.append(labels)
                # Monitor progress
                pbar.update(len(gal_loader) + i + 1)
            que_img_embs = torch.vstack(que_img_embs)
            que_img_lbls = torch.hstack(que_img_lbls)
        else:
            que_img_embs = None
            que_img_lbls = None
    
    pbar.finish()
    eval_res = img_ranking_evaluate_Market1501(
        gal_img_embs, gal_img_lbls, que_img_embs, que_img_lbls, export_result)

    logger.info("mAP: {:.03f} \t Score@1: {:.03f} \t Score@5: {:.03f} \t Score@10: {:.03f} \t Score@49: {:.03f}".format(
                eval_res["mAP"],
                eval_res["acc_top1"] * 100, 
                eval_res["acc_top5"] * 100, 
                eval_res["acc_top10"]* 100,   
                eval_res["acc_top49"]* 100
            ))
    return eval_res

if __name__ == "__main__":
    pass