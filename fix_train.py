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
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p

parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                    metavar='LR', help='initial learning rate')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')
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
parser.add_argument('--val-iteration', type=int, default=500,
                        help='Frequency for the evaluation')

# Dataset options
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                        help='Dataset')

# Hyperparameters for FixMatch
parser.add_argument('--tau', default=0.95, type=float, help='hyper-parameter for pseudo-label of FixMatch')
parser.add_argument('--ema-decay', default=0.999, type=float)

# Sampler options
parser.add_argument('--sampler', default='random', type=str, choices=['random', 'mean', 'reverse'],
                    help='Sampler for labeled data')
parser.add_argument('--semi-sampler', default='random', type=str, choices=['random', 'mean', 'reverse'],
                    help='Sampler for unlabeled data')
parser.add_argument('--pseudo-esp', default=1/3, type=float, help='Exponential hyper-parameter for unlabeled data')

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

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    print(f'==> Preparing imbalanced ' + args.dataset)

    N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, num_class, args.imb_ratio)
    U_SAMPLES_PER_CLASS = make_imb_data(args.ratio * args.num_max, num_class, args.imb_ratio)
    N_SAMPLES_PER_CLASS_T = torch.Tensor(N_SAMPLES_PER_CLASS)

    desi_p_m = torch.Tensor(U_SAMPLES_PER_CLASS).cuda()
    desi_p_m = desi_p_m / desi_p_m.max()
    desi_p_m = torch.flip(desi_p_m, [0])
    desi_p_r = desi_p_m.pow(2)
    desi_p_m = torch.pow(desi_p_m, args.pseudo_esp)
    desi_p_r = torch.pow(desi_p_r, args.pseudo_esp)
    desi_p_o = torch.ones(desi_p_m.shape).cuda()
    print('Sample ratio of random sampler:')
    print(desi_p_o)
    print('Sample ratio of mean sampler:')
    print(desi_p_m)
    print('Sample ratio of reverse sampler')
    print(desi_p_r)

    train_labeled_set, train_unlabeled_set, test_set = dataset.get_cifar('./data', N_SAMPLES_PER_CLASS, U_SAMPLES_PER_CLASS)

    target = torch.tensor(train_labeled_set.targets)
    class_sample_count = torch.tensor([(target == t).sum() for t in torch.unique(target, sorted=True)])
    mean_weight = 1. / class_sample_count.float()
    reverse_weight = mean_weight / class_sample_count.float()
    mean_samples_weight = torch.tensor([mean_weight[t] for t in target])
    reverse_samples_weight = torch.tensor([reverse_weight[t] for t in target])
    mean_sampler = data.WeightedRandomSampler(mean_samples_weight, len(mean_samples_weight))
    reverse_sampler = data.WeightedRandomSampler(reverse_samples_weight, len(reverse_samples_weight))
    if args.sampler == 'random':
        labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    elif args.sampler == 'mean':
        labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, sampler=mean_sampler, num_workers=4, drop_last=True)
    elif args.sampler == 'reverse':
        labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, sampler=reverse_sampler, num_workers=4, drop_last=True)

    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
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
    ema_model = create_model(ema=True)

    if use_cuda:
        cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ema_optimizer= WeightEMA(model, ema_model, alpha=args.ema_decay)
    start_epoch = 0

    # Resume
    title = 'fix-cifar'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss U', 'Test Loss', 'Test Acc'])

    test_accs = []

    if args.semi_sampler == 'random':
        desi_p = desi_p_o
    elif args.semi_sampler == 'mean':
        desi_p = desi_p_m
    elif args.semi_sampler == 'reverse':
        desi_p = desi_p_r

    # Main function
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        # Training part
        train_loss, train_loss_x, train_loss_u = train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer,
                                                       train_criterion, epoch, use_cuda, desi_p)

        # Evaluation part
        test_loss, test_acc, test_cls= validate(test_loader, ema_model, criterion, use_cuda, mode='Test Stats ')

        # Append logger file
        logger.append([train_loss, train_loss_x, train_loss_u, test_loss, test_acc])

        # Save models
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, epoch + 1, is_best=(test_acc > best_acc))

        if test_acc > best_acc:
            best_acc = test_acc

        test_accs.append(test_acc)

    logger.close()

    # Print the final results
    print('Mean bAcc:')
    print(np.mean(test_accs[-20:]))

    print('Best bAcc:')
    print(best_acc)

    print('Name of saved folder:')
    print(args.out)


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch, use_cuda, desi_p):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.val_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(args.val_iteration):
        try:
            inputs_x, targets_x, _ = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, _ = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2, inputs_u3), _, idx_u = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_u3), _, idx_u = unlabeled_train_iter.next()

        # Measure data loading time
        data_time.update(time.time() - end)
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, num_class).scatter_(1, targets_x.view(-1,1), 1)
        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u, inputs_u2, inputs_u3  = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda()

        # Generate the pseudo labels
        with torch.no_grad():
            # Generate the pseudo labels by aggregation and sharpening
            outputs_u, _ = model(inputs_u)
            targets_u = torch.softmax(outputs_u, dim=1)

        max_p, p_hat = torch.max(targets_u, dim=1)

        select_mask1 = max_p.ge(args.tau)
        select_mask2 = torch.rand(batch_size).cuda() < desi_p[p_hat]
        select_mask = select_mask1 * select_mask2
        select_mask = torch.cat([select_mask, select_mask], 0).float()

        p_hat = torch.zeros(batch_size, num_class).cuda().scatter_(1, p_hat.view(-1, 1), 1)

        all_inputs = torch.cat([inputs_x, inputs_u2, inputs_u3], dim=0)
        all_targets = torch.cat([targets_x, p_hat, p_hat], dim=0)

        all_outputs, _ = model(all_inputs)
        logits_x = all_outputs[:batch_size]
        logits_u = all_outputs[batch_size:]

        Lx, Lu = criterion(logits_x, all_targets[:batch_size], logits_u, all_targets[batch_size:], select_mask)
        loss = Lx + Lu

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                      'Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.val_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    )
        bar.next()
    bar.finish()

    return (losses.avg, losses_x.avg, losses_u.avg)

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

    classwise_correct = torch.zeros(num_class)
    classwise_num = torch.zeros(num_class)
    section_acc = torch.zeros(3)
    if use_cuda:
        classwise_correct = classwise_correct.cuda()
        classwise_num = classwise_num.cuda()
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

                classwise_correct[i] += (class_mask * pred_mask).sum()
                classwise_num[i] += class_mask.sum()

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

    # Major, Neutral, Minor
    section_num = int(num_class / 3)
    classwise_acc = (classwise_correct / classwise_num)
    section_acc[0] = classwise_acc[:section_num].mean()
    section_acc[2] = classwise_acc[-1 * section_num:].mean()
    section_acc[1] = classwise_acc[section_num:-1 * section_num].mean()

    if use_cuda:
        section_acc = section_acc.cpu()

    return (losses.avg, top1.avg, section_acc.numpy())

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

def save_checkpoint(state, epoch, checkpoint=args.out, filename='checkpoint.pth.tar', is_best=False):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if epoch % 100 == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_' + str(epoch) + '.pth.tar'))
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best_model.pth.tar'))

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, mask):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = -torch.mean(torch.sum(F.log_softmax(outputs_u, dim=1) * targets_u, dim=1) * mask)

        return Lx, Lu

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param = ema_param.float()
            param = param.float()
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param.mul_(1 - self.wd)

if __name__ == '__main__':
    main()
