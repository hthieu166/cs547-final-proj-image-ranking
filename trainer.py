"""Training routine"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import time
import ipdb
from torch.utils.tensorboard import SummaryWriter
import os.path as osp
from tester import test
from src.utils.misc import MiscUtils
import src.utils.logging as logging

logger = logging.get_logger(__name__)


def train(model, optimizer, criterion, loaders, logdir,
          train_mode, train_params, device, pretrained_model_path):
    """Training routine

    Args:
        model: model to train
        optimizer: optimizer to optimize the loss function
        criterion: loss function
        loaders: dictionary of data loader for training and validation
            'loaders[train]' for training
            'loaders[val]' for validationn
            ...
        logdir: where to store the logs
        train_mode: how to start the training. Only accept values as:
            'from_scratch': start the training from scratch
            'from_pretrained': start the training with a pretrained model
            'resume': resume an interrupted training
        train_params: training parameters as a dictionary
        device: id of the device for torch to allocate objects
    """
    # Setup training starting point
    if train_mode == 'from_scratch':
        start_epoch = 0
        lr = train_params['init_lr']
    elif train_mode == 'from_pretrained':
        model.load_model(pretrained_model_path)
        start_epoch = 0
        lr = train_params['init_lr']
    elif train_mode == 'resume':
        prefix = MiscUtils.get_lastest_checkpoint(logdir)
        lr, start_epoch = MiscUtils.load_progress(model, optimizer, prefix)
    else:
        raise ValueError('Unsupported train_mode: {}'.format(train_mode))

    # Set up some variables
    best_score = 0.0
    tensorboard_dir = osp.join(logdir, "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True )   
    writer = SummaryWriter(log_dir= tensorboard_dir)

    # Go through training epochs
    for epoch in range(start_epoch, train_params['n_epochs']):
        logger.info('epoch: %d/%d' % (epoch+1, train_params['n_epochs']))

        # Training phase
        train_loader = loaders['train']
        run_iter = epoch * len(train_loader)
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader,
                                     device, writer, run_iter)

        # Validation phase
        val_loader   = loaders["val"]
        img_ranking_acc = test(model, criterion, val_loader, device)
        # Log using Tensorboard
        writer.add_scalars('losses', {'train': train_loss}, epoch)
        writer.add_scalars('val_acc', img_ranking_acc, epoch)

        # Save training results when necessary
        if (epoch+1) % train_params['n_epochs_to_log'] == 0:
            MiscUtils.save_progress(model, optimizer, logdir, epoch)

        logger.info("Score@1: {:.03f} \t Score@10: {:.03f} \t Score@49: {:.03f}".format(
            img_ranking_acc["acc_top1"] *100, img_ranking_acc["acc_top10"] *100, img_ranking_acc["acc_top49"] * 100,  
        ))

        # Backup the best model
        val_score = img_ranking_acc["acc_top49"]
        if  val_score> best_score:
            logger.info('Current best score: %.2f' % val_score)
            best_score = val_score
            model.save_model(os.path.join(logdir, 'best.model'))

        # Decay learning Rate
        if epoch+1 in train_params['decay_epochs']:
            lr *= train_params['lr_decay']
            logger.info('Change learning rate to: %.5f' % lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                logger.info('%.5f' % param_group['lr'])

        logger.info('-'*80)
    writer.close()

def train_one_epoch(model, optimizer, criterion, train_loader, device, writer, run_iter):
    """Training routine for one epoch

    Args:
        model: model to train
        optimizer: optimizer to optimize the loss function
        criterion: loss function
        train_loader: data loader for training set
        device: id of the device for torch to allocate objects
        writer: summary writer to log training progress wrt some iterations
        run_iter: number of iterations already ran

    Return:
        train_loss: training loss of the epoch
    """
    # Switch to train mode
    model.train()

    # Set up progressbar
    pbar = MiscUtils.gen_pbar(max_value=len(train_loader), msg='Training: ')

    # Go through all samples of the training data
    train_loss, n_samples, start_time = 0.0, 0, time.time()
    for i, (samples, labels) in enumerate(train_loader):
        # Place data on the corresponding device
        samples = samples.to(device)
        labels = labels.to(device)

        # Forward + Backward + Optimize
        optimizer.zero_grad()

        a_feat  = model(samples[:,0,...])
        p_feat  = model(samples[:,1,...])
        n_feat  = model(samples[:,2,...])
        outputs = {'a_feat': a_feat, 'p_feat':p_feat, 'n_feat': n_feat}
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # Statistics
        n_samples += labels.size(0)
        train_loss += loss

        # Monitor the training progress
        pbar.update(i+1, loss=loss.item())
        run_iter += 1
        if run_iter % 100 == 0:
            writer.add_scalar('train_loss_per_iter', loss.item(), run_iter)
              
    pbar.finish()
    train_loss = train_loss.item()
    train_loss /= len(train_loader)

    logger.info('Training loss: %.4f' % train_loss)
    logger.info('Epoch running time: %.4fs' % (time.time()-start_time))

    return train_loss
