from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

import models.wrn as models
from utils import Bar, Logger, AverageMeter, accuracy

parser = argparse.ArgumentParser(description='PyTorch FixMatch Analysis')
# Optimization options
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
#Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

# Method options
parser.add_argument('--num_max', type=int, default=1500,
                        help='Number of samples in the maximal class')
parser.add_argument('--ratio', type=float, default=2.0,
                        help='Relative size between labeled and unlabeled data')
parser.add_argument('--imb_ratio', type=int, default=100,
                        help='Imbalance ratio for data')

# Dataset options
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                        help='Dataset')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

if args.dataset == 'cifar10':
    import dataset.fix_cifar10 as dataset
    num_class = 10
    args.num_max = 1500
elif args.dataset == 'cifar100':
    import dataset.fix_cifar100 as dataset
    num_class = 100
    args.num_max = 150

best_acc = 0  # best test accuracy

def main():
    global best_acc

    # Data
    print(f'==> Preparing imbalanced CIFAR-10')

    N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, num_class, args.imb_ratio)
    U_SAMPLES_PER_CLASS = make_imb_data(args.ratio * args.num_max, num_class, args.imb_ratio)
    N_SAMPLES_PER_CLASS_T = torch.Tensor(N_SAMPLES_PER_CLASS)

    _, _, test_set = dataset.get_cifar('./data', N_SAMPLES_PER_CLASS, U_SAMPLES_PER_CLASS)

    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    print("==> creating WRN-28-2")

    def create_model(ema=False):
        model = models.WRN(2, num_class)
        if use_cuda:
            model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()

    if use_cuda:
        cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()

    # Resume
    title = 'fix-cifar'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['ema_state_dict'])

    # Evaluation part
    test_loss, test_acc, test_cls, test_classwise_num, test_classwise_precision, test_classwise_recall = validate(test_loader, model, criterion, use_cuda, mode='Test Stats ')

    print('Mean acc:')
    print(test_acc)
    print('Per-class precision:')
    print(test_classwise_precision)
    print('Per-class recall')
    print(test_classwise_recall)


def validate(valloader, model, criterion, use_cuda, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))

    classwise_num = torch.zeros(num_class)
    classwise_TP = torch.zeros(num_class)
    classwise_FP = torch.zeros(num_class)
    section_acc = torch.zeros(3)
    if use_cuda:
        classwise_num = classwise_num.cuda()
        classwise_TP = classwise_TP.cuda()
        classwise_FP = classwise_FP.cuda()
        section_acc = section_acc.cuda()

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # classwise prediction
            pred_label = outputs.max(1)[1]
            pred_mask = (targets == pred_label).float()
            for i in range(num_class):
                class_mask = (targets == i).float()
                classwise_num[i] += class_mask.sum()
                classwise_TP[i] += (class_mask * pred_mask).sum()
                classwise_FP[i] += ((1 - class_mask) * ((pred_label == i).float())).sum()

             # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                          'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()

    print(classwise_FP)
    # Major, Neutral, Minor
    section_num = int(num_class / 3)
    classwise_precision = (classwise_TP / (classwise_TP + classwise_FP))
    classwise_recall = (classwise_TP / classwise_num)
    section_acc[0] = classwise_recall[:section_num].mean()
    section_acc[2] = classwise_recall[-1 * section_num:].mean()
    section_acc[1] = classwise_recall[section_num:-1 * section_num].mean()

    if use_cuda:
        classwise_num = classwise_num.cpu()
        classwise_precision = classwise_precision.cpu()
        classwise_recall = classwise_recall.cpu()
        section_acc = section_acc.cpu()

    return (losses.avg, top1.avg, section_acc.numpy(), classwise_num.numpy(), classwise_precision.numpy(), classwise_recall.numpy())

def make_imb_data(max_num, class_num, gamma):
    mu = np.power(1/gamma, 1/(class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / gamma))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))
    print(class_num_list)
    return list(class_num_list)

if __name__ == '__main__':
    main()
