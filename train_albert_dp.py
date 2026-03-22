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
import pandas as pd
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

import wandb

DATA_ROOT = './data/amazon_review_full_csv'

class AmazonReviewsDataset(torch.utils.data.Dataset):

    def __init__(self, split, maxlen, bert_model='albert-base-v2'):

        self.split = split  # pandas dataframe
        #Initialize the tokenizer
        self.data = pd.read_csv(os.path.join(DATA_ROOT, f'{split}.csv'), names=['class', 'title', 'review'], dtype={'class':int, 'title':str, 'review':str})
        self.data['x'] = self.data['title'] + ' ' + self.data['review']
        self.data['y'] = self.data['class'].apply(lambda x: int(x-1))  # Convert to 0-indexed labels
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)

        self.maxlen = maxlen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Selecting sentence1 and sentence2 at the specified index in the data frame
        x = str(self.data.loc[index, 'x'])

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(x,
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=self.maxlen,  
                                      return_tensors='pt')  # Return torch.Tensor objects
        
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        label = self.data.loc[index, 'y']
        return token_ids, attn_masks, token_type_ids, label

class ALBERT(nn.Module):
    def __init__(self, bert_model='albert-base-v2', freeze=False):
        super(ALBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        self.bert_model = bert_model
        self.freeze = freeze
        if self.freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        # cont_reps, pooler_out = self.bert(input_ids, attention_mask, token_type_ids)
        bert_out = self.bert(**x)
        pooler_out = bert_out['pooler_output']
        return self.tanh(pooler_out)

class ALBERTClassifier(nn.Module):
    def __init__(self, bert_model='albert-base-v2', num_classes=5):
        super(ALBERTClassifier, self).__init__()
        if bert_model == "albert-base-v2":  # 12M parameters
            self.hidden_size = 768
        elif bert_model == "albert-large-v2":  # 18M parameters
            self.hidden_size = 1024
        elif bert_model == "albert-xlarge-v2":  # 60M parameters
            self.hidden_size = 2048
        elif bert_model == "albert-xxlarge-v2":  # 235M parameters
            self.hidden_size = 4096
        elif bert_model == "bert-base-uncased": # 110M parameters
            self.hidden_size = 768
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        x = self.classifier(x)
        return x

parser = argparse.ArgumentParser(description='Split Learning ALBERT Base for Amazon Reviews')
parser.add_argument('--arch', '-a', metavar='ARCH', default='albert-base-v2',
                    # choices=model_names,
                    # help='model architecture: ' + ' | '.join(model_names) +
                    # ' (default: resnet32)'
                    )
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                    metavar='LR', help='initial learning rate (default: 5e-5)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-2, type=float,
                    metavar='W', help='weight decay (default: 1e-2)')
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
                    type=int, default=1)
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

parser.add_argument('--run-name', type=str, default=None,
                    help='run name for wandb')
parser.add_argument('--run-id', type=str, default=None,
                    help='run id for wandb (only used to resume the run)')


class ReShaper(nn.Module):
    def __init__(self):
        super(ReShaper, self).__init__()
    def forward(self, x):
        return x.squeeze()

best_prec1 = 0

# def resnet20_sp():
#     return ResNet_SP(BasicBlock_DP, [3, 3, 3])

def main():
    global args, best_prec1
    args = parser.parse_args()
    # args = parser.parse_args(args=[])


    # args.split_layer = -1
    # args.enable_dp = True
    # args.sigma = 0.7

    # args.enable_denoise = True
    # args.scaling_factor = 1.0
    # args.mask_ratio = 0.2
    # args.avg_count = 1
    # args.weight_decay = 0.0

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # Check if the resume path exists or not
    else:
        if os.path.exists(os.path.join(args.save_dir, 'checkpoint.pth')) and os.path.exists(os.path.join(args.save_dir, 'best_model.pth')):
            # print('Checkpoint and best model already exists.')
            if not args.evaluate and args.resume == '':
                print('Found existing checkpoint and manually overriding resume. Resuming from the found checkpoint instead of starting a new run')
                temp = torch.load(os.path.join(args.save_dir, 'checkpoint.pth'))
                wandb_id = temp['wandb_id']
                del temp
                args.resume = os.path.join(args.save_dir, 'checkpoint.pth')
                if args.run_name is not None:
                    if wandb_id is None: raise ValueError('Run ID not found in the disovered checkpoint!')
                    args.run_id = wandb_id
                    # print('Resuming from run id: {}'.format(wandb_id))

    if args.run_name is not None:
        if not args.evaluate and len(args.resume)>0:
            if args.run_id is not None:
                print('Resuming the existing run...')
                wandb.init(entity='AccurateSplitLearning', project='SplitLeaning', name=args.run_name, id=args.run_id, resume='must', config=args, config_exclude_keys=['run_name', 'run_id'])
            else:
                raise ValueError('run_id must be specified when resuming a run')
        else:
            print('Starting a new run...')
            wandb.init(entity='AccurateSplitLearning', project='SplitLeaning', name=args.run_name, config=args, config_exclude_keys=['run_name', 'run_id'])

    # model = torch.nn.DataParallel(resnet20())
    # model = resnet20_sp()
    model = nn.Sequential(
        ALBERT(bert_model='albert-base-v2', freeze=False),
        nn.Dropout(0.1),
        ALBERTClassifier(bert_model='albert-base-v2', num_classes=5)
    )
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
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    train_loader = torch.utils.data.DataLoader(
        AmazonReviewsDataset(split='train', maxlen=128, bert_model='albert-base-v2'),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        AmazonReviewsDataset(split='test', maxlen=128, bert_model='albert-base-v2'),
        batch_size=256, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                     milestones=[100, 150], last_epoch=args.start_epoch - 1)

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
    # for m in models:
    #     m.apply(_weights_init)
    print(models)
    client_opt = torch.optim.AdamW(client_model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
    server_opt = torch.optim.AdamW(server_model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
    optimizers = client_opt, server_opt
    
    if args.resume and not args.evaluate:
        client_opt.load_state_dict(checkpoint['optimizer']['client'])
        server_opt.load_state_dict(checkpoint['optimizer']['server'])
    
    client_scheduler = get_linear_schedule_with_warmup(optimizer=client_opt, num_warmup_steps=0, num_training_steps=int(len(train_loader)*args.epochs))
    server_scheduler = get_linear_schedule_with_warmup(optimizer=server_opt, num_warmup_steps=0, num_training_steps=int(len(train_loader)*args.epochs))

    lr_schedulers = client_scheduler, server_scheduler

    # if args.arch in ['resnet1202', 'resnet110']:
    #     # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
    #     # then switch back. In this setup it will correspond for first epoch.
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = args.lr*0.1

    if args.resume and not args.evaluate:
        # client_opt.load_state_dict(checkpoint['optimizer']['client'])
        # server_opt.load_state_dict(checkpoint['optimizer']['server'])
        client_scheduler.load_state_dict(checkpoint['lr_scheduler']['client'])
        server_scheduler.load_state_dict(checkpoint['lr_scheduler']['server'])


    if args.evaluate:
        validate(args, val_loader, models, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizers[0].param_groups[0]['lr']))
        train_prec1, train_loss = train(args, train_loader, val_loader, models, criterion, optimizers, lr_schedulers, epoch)
        # lr_scheduler.step()
        # client_scheduler.step()
        # server_scheduler.step()

        # evaluate on validation set
        prec1, test_loss = validate(args, val_loader, models, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print(f'Epoch: {epoch}, best_prec1: {best_prec1}')

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': {'client': client_opt.state_dict(), 'server': server_opt.state_dict()},
                'lr_scheduler': {'client': client_scheduler.state_dict(), 'server': server_scheduler.state_dict()},
                'wandb_id': wandb.run.id if args.run_name is not None else None,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.pth'))

        if is_best:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'best_model.pth'))
        
        if args.run_name is not None:
            wandb.log({
                'train_loss': train_loss, 'train_acc': train_prec1, 'test_loss': test_loss, 'test_acc': prec1,
                'best_acc': best_prec1, 'epoch': epoch
            }, step=epoch)


def train(args, train_loader, val_loader, models, criterion, optimizers, lr_schedulers, epoch):
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
    for i, (token_ids, attn_masks, token_type_ids, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = {'input_ids': token_ids.cuda(), 'attention_mask': attn_masks.cuda(), 'token_type_ids': token_type_ids.cuda()}
        target_var = target
        if args.half:
            # input_var = input_var.half()
            for k, v in input_var.items():
                input_var[k] = v.half()

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

        # if args.weight_decay:
        #     for model in models:
        #         for name,p in model.named_parameters():
        #             p.grad.data.add_(p.data, alpha=-args.weight_decay)

        # compute gradient and do SGD step
        for opt in optimizers:
            opt.step()

        for sch in lr_schedulers:
            sch.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), target.size(0))
        top1.update(prec1.item(), target.size(0))

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
    
    return top1.avg, losses.avg


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
        for i, (token_ids, attn_masks, token_type_ids, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = {'input_ids': token_ids.cuda(), 'attention_mask': attn_masks.cuda(), 'token_type_ids': token_type_ids.cuda()}
            target_var = target.cuda()

            if args.half:
                # input_var = input_var.half()
                for k, v in input_var.items():
                    input_var[k] = v.half()

            # compute output

            # output = model(input_var)
            output = server_model(client_model(input_var))
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), target.size(0))
            top1.update(prec1.item(), target.size(0))

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

    return top1.avg, losses.avg

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


# main()
if __name__ == '__main__':
    main()