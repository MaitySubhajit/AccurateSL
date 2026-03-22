import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.distributions.normal import Normal

from resnet import BasicBlock, BasicBlock_DP, LambdaLayer, _weights_init, ResNet_SP


parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    # choices=model_names,
                    # help='model architecture: ' + ' | '.join(model_names) +
                    # ' (default: resnet32)'
                    )
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--split-layer', type=int, default=-1, metavar='N',
                    help='index of the layer to split, usually we split after activation layer (default: 9)')
parser.add_argument('--enable-dp', action='store_true', default=False,
                    help='add dp gaussian noise on tensors trasmitted from party A to party B')
parser.add_argument('--sigma', type=float, default=0.7, metavar='M',
                    help='Std of the Gaussian noise (default: 0.7)')
parser.add_argument('--enable-denoise', action='store_true', default=False,
                    help='whether to use denoise methods, e.g. scaling and dropout')
parser.add_argument('--scaling-factor', type=float, default=1.0, metavar='M',
                    help='scale the value of noise injected tensors')
parser.add_argument('--mask-ratio', type=float, default=1.0, metavar='M',
                    help='add mask on noise injected tensors')
parser.add_argument('--avg-count', type=int, default=1, metavar='N',
                    help='averaging counts for droupout')

best_prec1 = 0


def resnet20_sp():
    return ResNet_SP(BasicBlock_DP, [3, 3, 3])

def main():
    global args, best_prec1
    # args = parser.parse_args()
    args = parser.parse_args(args=[])


    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # model = torch.nn.DataParallel(resnet20())
    model = resnet20_sp()
    model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../cifar10', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../cifar10', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    args.dropout_ratio = 1 - args.mask_ratio
    args.best_acc = []

    print("All configurations:\n", args)
    if args.enable_dp:
        print(f"dp is enabled! Std of the Gaussian noise: {args.sigma}")
    if args.weight_decay:
        print("weight_decay is enabled! decay parameter:", args.weight_decay)
    if args.enable_denoise:
        if args.scaling_factor != 1.0:
            print(f'scaling is enabled! scaling factor: {args.scaling_factor}')
        if args.dropout_ratio:
            print(f'dropout is enabled! ratio: {args.dropout_ratio}, averaging count: {args.avg_count}')
        else:
            args.avg_count = 1

    client_model = nn.Sequential(*nn.ModuleList(model.children())[:args.split_layer])
    server_model = nn.Sequential(*nn.ModuleList(model.children())[args.split_layer:])
    models = client_model, server_model
    for m in models:
        m.apply(_weights_init)
    print(models)
    client_opt = torch.optim.SGD(client_model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    server_opt = torch.optim.SGD(server_model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    optimizers = client_opt, server_opt
    client_scheduler = torch.optim.lr_scheduler.MultiStepLR(client_opt,
                                                        milestones=[100, 150], last_epoch=args.start_epoch - 1)
    server_scheduler = torch.optim.lr_scheduler.MultiStepLR(server_opt,
                                                        milestones=[100, 150], last_epoch=args.start_epoch - 1)

    if args.evaluate:
        validate(val_loader, models, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizers[0].param_groups[0]['lr']))
        train(args, train_loader, val_loader, models, criterion, optimizers, epoch)
        # lr_scheduler.step()
        client_scheduler.step()
        server_scheduler.step()

        # evaluate on validation set
        prec1 = validate(args, val_loader, models, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print(f'Epoch: {epoch}, best_prec1: {best_prec1}')


def train(args, train_loader, val_loader, models, criterion, optimizers, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    client_model, server_model = models
    client_opt, server_opt = optimizers
    # switch to train mode
    for model in models:
        model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        for opt in optimizers:
            opt.zero_grad()
        # compute output
        # output = model(input_var)
        # loss = criterion(output, target_var)
        client_acts = client_model(input_var)
        server_acts = torch.empty_like(client_acts, requires_grad=True)
        server_acts.data = client_acts.data
        if args.enable_dp:
            size = server_acts.size()
            noise = Normal(0.0, args.sigma).sample(sample_shape=size)
            server_acts.data.add_(noise.to(server_acts.device))
        server_acts.retain_grad()
        if args.enable_denoise:
            acts = server_acts.data.clone().detach()
            n = args.avg_count
            for _ in range(n):
                drop = nn.Dropout(args.dropout_ratio, inplace=False)
                server_acts.data = drop(acts.data) * (1-args.dropout_ratio)
                output = server_model(server_acts)
                output.data *= args.scaling_factor
                loss = criterion(output, target_var)
                loss.backward()

            server_acts.grad.data.div_(n)
            for name,p in server_model.named_parameters():
                p.grad.data.div_(n)
        else:
            output = server_model(server_acts)
            loss = criterion(output, target_var)
            loss.backward()

        client_acts.grad = server_acts.grad
        client_acts.backward(client_acts.grad)

        # compute gradient and do SGD step
        for opt in optimizers:
            opt.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(args, val_loader, models, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    for model in models:
        model.eval()

    end = time.time()
    client_model, server_model = models
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output

            # output = model(input_var)
            output = server_model(client_model(input_var))
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


main()
# if __name__ == '__main__':
#     main()