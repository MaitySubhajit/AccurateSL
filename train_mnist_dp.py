# bash script export
# refer to https://github.com/pytorch/examples/blob/main/mnist/main.py
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions.laplace import Laplace
import os


class Reshape(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.permute(x, (0, 2, 3, 1))
        return x

    def name(self):
        return "swap axies"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.act1 = nn.Tanh()
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.act2 = nn.Tanh()
        self.maxpoole2d = nn.MaxPool2d(2)

        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(64*14*14, 256)
        self.act_fc1 = nn.Tanh()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.maxpoole2d(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act_fc1(x)
        x = self.fc2(x)

        return x


def train(args, models, device, train_loader, test_loader, optimizers, epoch):
    client_model, server_model = models
    client_opt, server_opt = optimizers
    for model in models:
        model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        for opt in optimizers:
            opt.zero_grad()

        client_acts = client_model(data)
        if args.add_noise:
            size = client_acts.size()
            noise = Normal(0.0, args.sigma).sample(sample_shape=size)
            client_acts.data.add_(noise.to(client_acts.device))

        server_acts = torch.zeros_like(client_acts, requires_grad=True)
        server_acts.data = client_acts.data
        
        if args.enable_denoise:
            drop = nn.Dropout(1 - args.mask_ratio, inplace=False)
            if args.dropout_only:
                server_acts.data = drop(server_acts.data) 
            else:
                server_acts.data = (drop(server_acts.data) * (args.mask_ratio))
            server_acts.data *= args.scaling_factor
            output = server_model(server_acts)
            # output.data *= args.scaling_factor
            output = nn.LogSoftmax(dim=1)(output)
            loss = F.nll_loss(output, target)
            server_acts.data /= args.scaling_factor
            loss.backward()
            
        else:
            output = server_model(server_acts)
            output = nn.LogSoftmax(dim=1)(output)
            loss = F.nll_loss(output, target)
            loss.backward()

        client_acts.grad = server_acts.grad
        client_acts.backward(client_acts.grad)

        if args.weight_decay:
            for model in models:
                for name,p in model.named_parameters():
                    p.grad.data.add_(p.data, alpha=args.weight_decay)

        client_opt.step()
        server_opt.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} Step: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, args.step, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
        if batch_idx % args.test_interval == 0:
            test(args, models, device, test_loader, epoch)
            for model in models:
                model.train()
        args.step += 1


def test(args, models, device, test_loader, epoch):
    for model in models:
        model.eval()
    client_model, server_model = models
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            client_acts = client_model(data)
            server_acts = client_acts.clone().detach()
            output = server_model(server_acts)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set Accuracy: {}/{} ({:.2f}%)'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    args.best_acc = max(correct / len(test_loader.dataset), args.best_acc)
    print(f'Best accuracy: {args.best_acc}\n')
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before test')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--split-layer', type=int, default=-1, metavar='N',
                        help='index of the layer to split, usually we split after activation layer (default: -1)')
    parser.add_argument('--add-noise', action='store_true', default=False,
                        help='add gaussian noise on tensors trasmitted from the client to the server')
    parser.add_argument('--sigma', type=float, default=0.7, metavar='M',
                        help='Std of the Gaussian noise (default: 0.7)')
    parser.add_argument('--enable-denoise', action='store_true', default=False,
                        help='whether to use denoising methods, e.g. scaling and masking')
    parser.add_argument('--dropout-only', action='store_true', default=False,
                        help='whether to use dropout to replace masking operation')
    parser.add_argument('--scaling-factor', type=float, default=1.0, metavar='M',
                        help='multiply the noise injected tensors with scaling_factor')
    parser.add_argument('--mask-ratio', type=float, default=1.0, metavar='M',
                        help='the ratio of elements kept after masking')
    parser.add_argument('--weight-decay', type=float, default=0.0, metavar='M',
                        help='weight decay factor')
    args = parser.parse_args()


    print("All configurations:\n", args)
    if args.add_noise:
        print(f"noise injection is enabled! Std of the Gaussian noise: {args.sigma}")
    if args.weight_decay:
        print("weight_decay is enabled! decay parameter:", args.weight_decay)
    if args.enable_denoise:
        if args.scaling_factor != 1.0:
            print(f'scaling is enabled! scaling factor: {args.scaling_factor}')
        if args.mask_ratio != 1.0:
            print(f'masking is enabled! ratio: {args.mask_ratio},')
        
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)

    for images, labels in train_loader:  
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        break
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        out = model(data)
        break

    client_model = nn.Sequential(*nn.ModuleList(model.children())[:args.split_layer])
    server_model = nn.Sequential(*nn.ModuleList(model.children())[args.split_layer:])
    client_opt = optim.SGD(client_model.parameters(), lr=args.lr)
    server_opt = optim.SGD(server_model.parameters(), lr=args.lr)
    client_scheduler = StepLR(client_opt, step_size=1, gamma=args.gamma)
    server_scheduler = StepLR(server_opt, step_size=1, gamma=args.gamma)

    models = client_model, server_model
    print("\nSplit learning configurations:")
    print(f"split the model at layer {args.split_layer}")
    print(models)
    opts = client_opt, server_opt
    args.step = 0
    args.best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train(args, models, device, train_loader, test_loader, opts, epoch)
        client_scheduler.step()
        server_scheduler.step()

if __name__ == '__main__':
    main()

