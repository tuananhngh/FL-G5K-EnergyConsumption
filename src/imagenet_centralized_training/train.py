#!/usr/bin/env python3
#
# Note -- this training script is tweaked from the original at:
#           https://github.com/pytorch/examples/tree/master/imagenet
#
# For a step-by-step guide to transfer learning with PyTorch, see:
#           https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
#
import argparse
import os
import logging
import random

import time
from tqdm import tqdm
import shutil
import warnings
import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('/home/mjay/jetson-inference/python/training/classification')
from reshape import reshape_model

from datasets import load_from_disk
from utils.dataset import MyDataset
import numpy as np

# get the available network architectures
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
global step
step = 0

def parse_args():
    # parse command-line arguments
    parser = argparse.ArgumentParser(description='PyTorch Image Classifier Training')

    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--dataset-type', type=str, default='folder',
                        choices=['folder', 'nuswide', 'voc'],
                        help='specify the dataset type (default: folder)')
    parser.add_argument('--multi-label', action='store_true',
                        help='multi-label model (aka image tagging)')
    parser.add_argument('--multi-label-threshold', type=float, default=0.5,
                        help='confidence threshold for counting a prediction as correct')
    parser.add_argument('--model-dir', type=str, default='models', 
                        help='path to desired output directory for saving model '
                        'checkpoints (default: models/)')
    parser.add_argument('--log-dir', type=str, default='logs', 
                        help='path to desired output directory for saving model '
                        'checkpoints (default: logs/)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
    parser.add_argument('--resolution', default=224, type=int, metavar='N',
                        help='input NxN image resolution of model (default: 224x224) '
                            'note than Inception models should use 299x299')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=35, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N',
                        help='mini-batch size (default: 8)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-v', '--eval-freq', default=1000, type=int,
                        metavar='N', help='Evaluation frequency (default: 1000)')
    parser.add_argument('--early-stop', default=3000, type=int,
                        metavar='N', help='Number of steps of not improving validation acc before stopping training (default: 3000)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU ID to use (default: 0)')

            
    return parser.parse_args()

def main(args):
    """
    Load dataset, setup model, and train for N epochs
    """
    global best_accuracy
    global step
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        logging.info(f"=> using GPU {args.gpu} ({torch.cuda.get_device_name(args.gpu)})")

    # load the dataset
    ds = load_from_disk(args.data)
    logging.info("dataset loaded")
    train_dataset = MyDataset(ds, 'train', args.resolution)
    val_dataset = MyDataset(ds, 'validation', args.resolution)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset , 
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # create or load the model if using pre-trained (the default)
    if args.pretrained:
        logging.info(f"=> using pre-trained model '{args.arch}'")
        model = models.__dict__[args.arch](pretrained=True)
    else:
        logging.info(f"=> creating model '{args.arch}'")
        model = models.__dict__[args.arch]()

    # reshape the model for the number of classes in the dataset
    model = reshape_model(model, args.arch, 1000)

    # define loss function (criterion) and optimizer
    if args.multi_label:
        criterion = nn.NLLLoss() #nn.BCEWithLogitsLoss() # NLLLoss
    else:
        criterion = nn.CrossEntropyLoss()

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    tensorboard.add_scalar('lr', args.lr, step)
        
    # transfer the model to the GPU that it should be run on
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        criterion = criterion.cuda(args.gpu)
        torch.cuda.empty_cache()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            # best_accuracy = checkpoint['best_accuracy']
            # if args.gpu is not None:
            #     best_accuracy = best_accuracy.to(args.gpu)   # best_accuracy may be from a checkpoint from a different GPU
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            logging.info(f"=> no checkpoint found at '{args.resume}'")

    cudnn.benchmark = True

    # if in evaluation mode, only run validation
    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    # get the start time
    start = time.time()
    logging.info(f"Starting training: {start}")
    tensorboard.add_scalar('Timestamp_start/train', start, step)
    
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train_loss, train_acc, train_time, val_loss, val_acc, val_time = train(train_loader, val_loader, model, criterion, optimizer)

        if train_loss is None:
            break
        # evaluate on validation set
        # val_loss, val_acc, val_time = validate(val_loader, model, criterion, epoch)

        # remember best acc@1 and save checkpoint
        is_best = val_acc > best_accuracy
        best_accuracy = max(val_acc, best_accuracy)

        logging.info(f"=> Epoch {epoch}")
        logging.info(f"  * Train Loss     {train_loss:.4e}")
        logging.info(f"  * Train Accuracy {train_acc:.4f}")
        logging.info(f"  * Train time     {train_time:.4f}")
        logging.info(f"  * Val time       {val_time:.4e}")
        logging.info(f"  * Val Loss       {val_loss:.4e}")
        logging.info(f"  * Val Accuracy   {val_acc:.4f}{'*' if is_best else ''}")
        
        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'resolution': args.resolution,
            'classes': "check list of classes in the dataset folder",
            'num_classes': 1000,
            'multi_label': args.multi_label,
            'state_dict': model.state_dict(),
            'accuracy': {'train': train_acc, 'val': val_acc},
            'loss' : {'train': train_loss, 'val': val_loss},
            'optimizer' : optimizer.state_dict(),
            'best_accuracy' : best_accuracy,
        }, is_best)
        
        torch.cuda.empty_cache()


def train(train_loader, val_loader, model, criterion, optimizer):
    """
    Train one epoch over the dataset
    """
    global step
    
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Accuracy', ':7.3f')
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc], 
        prefix=f"Step: [{step}]")

    # switch to train mode
    model.train()

    end = time.time()
    
    # with tqdm(train_loader, unit="batch") as tepoch:
    progress_bar = tqdm(train_loader, unit="batch", file=open(args.log_dir+'train.log', 'w'))
    for i, (images, target) in enumerate(progress_bar):
        # step limit
        if step > args.epochs:
            return None, None, None, None, None, None
        
        # decay the learning rate
        adjust_learning_rate(optimizer, step)
        
        # measure data loading time
        logging.info(str(progress_bar))
        step += 1
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
                
        # compute output
        output = model(images)
        loss = criterion(output, target)

        # record loss and measure accuracy
        losses.update(loss.item(), images.size(0))
        acc_inst = accuracy(output, target)
        acc.update(acc_inst, images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time= time.time() - end
        end = time.time()

        if i % args.print_freq == 0 or i == len(train_loader)-1:
            progress.display(i)
            
            step_timestamp = time.time()
            tensorboard.add_scalar('Timestamp_step/train', step_timestamp, step)
            tensorboard.add_scalar('Loss/train', loss, step)
            tensorboard.add_scalar('Accuracy/train', acc_inst, step)
            tensorboard.add_scalar('Time/train', batch_time, step)
            batch_time = AverageMeter('Time', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            acc = AverageMeter('Accuracy', ':7.3f')
            
        if not torch.isfinite(loss).all():
            logging.info("Loss is not finite, stopping training")
            return None, None, None, None, None, None
            
        if i % args.eval_freq == 0 or i == len(train_loader)-1:
            
            val_loss, val_acc, val_time = validate(val_loader, model, criterion, step)
            
            global best_accuracy
            global last_best_accuracy_step
            if val_acc > best_accuracy:
                last_best_accuracy_step = step
                best_accuracy = max(val_acc, best_accuracy)
                tensorboard.add_scalar('Best_accuracy/val', best_accuracy, i)
            else:
                if last_best_accuracy_step > step:
                    dist = len(train_loader) - last_best_accuracy_step + step
                else:
                    dist = step - last_best_accuracy_step
                if dist > args.early_stop:
                    logging.info("No improvement for %s steps, stopping training", args.early_stop)
                    return None, None, None, None, None, None

    return losses.avg, acc.avg, batch_time.avg, val_loss, val_acc, val_time
    

def validate(val_loader, model, criterion, step):
    """
    Measure model performance across the val dataset
    """
    torch.cuda.empty_cache()
    
    val_batch_time = AverageMeter('Time', ':6.3f')
    val_losses = AverageMeter('Loss', ':.4e')
    val_acc = AverageMeter('Accuracy', ':7.3f')
    
    start_val = time.time()
    
    progress = ProgressMeter(
        len(val_loader),
        [val_batch_time, val_losses, val_acc],
        prefix='Val:   ')

    # switch to evaluate mode
    model.eval()
    
    epoch_start = time.time()
    logging.info(f"Validation timestamp: {epoch_start}")
    tensorboard.add_scalar('Timestamp/val', epoch_start, step)

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            val_loss = criterion(output, target)

            # record loss and measure accuracy
            val_losses.update(val_loss.item(), images.size(0))
            val_acc.update(accuracy(output, target), images.size(0))
            
            # measure elapsed time
            val_batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or i == len(val_loader)-1:
                progress.display(i)
                
    val_end = time.time()
    val_duration = val_end - start_val 
    tensorboard.add_scalar('Timestamp_end/val', val_end, step)

    tensorboard.add_scalar('Loss/val', val_losses.avg, step)
    tensorboard.add_scalar('Accuracy/val', val_acc.avg, step)
    tensorboard.add_scalar('Time/val', val_batch_time.avg, step)
    tensorboard.add_scalar('Duration/val', val_duration, step)
    
    return val_losses.avg, val_acc.avg, val_batch_time.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar', labels_filename='labels.txt'):
    """
    Save a model checkpoint file, along with the best-performing model if applicable
    """
    if args.model_dir:
        model_dir = os.path.expanduser(args.model_dir)

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        filename = os.path.join(model_dir, filename)
        best_filename = os.path.join(model_dir, best_filename)
        # labels_filename = os.path.join(model_dir, labels_filename)
        
    # save the checkpoint
    torch.save(state, filename)
            
    # earmark the best checkpoint
    if is_best:
        shutil.copyfile(filename, best_filename)
        logging.info(f"saved best model to:  {best_filename}")
    else:
        logging.info(f"saved checkpoint to:  {filename}")
        
    # save labels.txt on the first epoch
    # if state['epoch'] == 0:
    #     with open(labels_filename, 'w') as file:
    #         for label in state['classes']:
    #             file.write(f"{label}\n")
    #     logging.info(f"saved class labels to:  {labels_filename}")
            

def adjust_learning_rate(optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    global step
    lr = args.lr * (0.1 ** (step // 20000))
    tensorboard.add_scalar('lr', lr, step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    """
    Computes the accuracy of predictions vs groundtruth
    """
    with torch.no_grad():
        if args.multi_label:
            output = F.sigmoid(output)
            preds = ((output >= args.multi_label_threshold) == target.bool())   # https://medium.com/@yrodriguezmd/tackling-the-accuracy-multi-metric-9e2356f62513
            
            # https://stackoverflow.com/a/61585551
            #output[output >= args.multi_label_threshold] = 1
            #output[output < args.multi_label_threshold] = 0
            #preds = (output == target)
        else:
            output = F.softmax(output, dim=-1)
            _, preds = torch.max(output, dim=-1)
            preds = (preds == target)
            
        return preds.float().mean().cpu().item() * 100.0
        
        
class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """
    Progress metering
    """
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    args = parse_args()
    
    open(args.log_dir+'train.log', mode='a').close()
    logging.basicConfig(
        filename=args.log_dir + "/train.log", 
        level=logging.INFO,
        format='%(levelname)s - %(asctime)s - %(filename)s - %(lineno)d : %(message)s',
        )
    
    logging.info("Training args: %s", args.__dict__)
    
    # open tensorboard logger (to model_dir/tensorboard)
    tensorboard = SummaryWriter(log_dir=args.log_dir)
    logging.info(f"To start tensorboard run:  tensorboard --log-dir={args.log_dir}")
    
    # variable for storing the best model accuracy so far
    best_accuracy = 0
    last_best_accuracy_step = 0
    
    main(args)
    # logging.info("Train.py executed")
