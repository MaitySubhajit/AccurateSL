#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Training sentiment prediction model on IMDB movie reviews dataset.
Architecture and reference results from https://arxiv.org/pdf/1911.11607.pdf
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from opacus import PrivacyEngine
from torch.functional import F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizerFast
from torch.distributions.normal import Normal


# TODO: this is still broken

# 1 torch.Size([64, 256])
# 2 torch.Size([64, 256, 16])
# 3 torch.Size([64, 16, 256])
# 4 torch.Size([64, 16])
# 5 torch.Size([64, 16])
# 6 torch.Size([64, 2])

# emb.weight torch.Size([256, 16])
# fc1.weight torch.Size([16, 16])
# fc1.bias torch.Size([16])
# fc2.weight torch.Size([2, 16])
# fc2.bias torch.Size([2])

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1, self.dim2 = dim1, dim2
    def forward(self, x):
        x = x.transpose(self.dim1, self.dim2)
        return x

    def name(self):
        return f"Transpose({self.dim1},{self.dim2})"


class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.squeeze()

    def name(self):
        return f"Squeeze()"


class SampleNet(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 16)
        self.transpose = Transpose(1, 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.squeeze = Squeeze()
        self.act1 = nn.Tanh()
        self.fc1 = nn.Linear(16, 16)
        self.act2 = nn.Tanh()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.emb(x)
        x = self.transpose(x)
        x = self.pool(x)
        x = self.squeeze(x)
        x = self.act1(x)
        x = self.fc1(x)
        x = self.act2(x)  # relu
        x = self.fc2(x)
        return x

    def name(self):
        return "SampleNet"


def binary_accuracy(preds, y):
    correct = (y.long() == torch.argmax(preds, dim=1)).float()
    acc = correct.sum() / len(correct)
    return acc


def padded_collate(batch, padding_idx=0):
    x = pad_sequence(
        [elem["input_ids"] for elem in batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    y = torch.stack([elem["label"] for elem in batch]).long()
    return x, y


def train(args, model, train_loader, test_loader, optimizer, privacy_engine, epoch):
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    device = torch.device(args.device)
    model = model.train()

    for data, label in (train_loader):
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        predictions = model(data).squeeze(1)
        loss = criterion(predictions, label)
        acc = binary_accuracy(predictions, label)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accuracies.append(acc.item())
        # print(
        #     f"Train Step: {args.step}\t Loss: {np.mean(losses):.6f}\t Accuracy: {np.mean(accuracies):.6f}"
        # )
        if args.step % 100 == 0:
            evaluate(args, model, test_loader)
            model = model.train()
        args.step += 1
    if not args.disable_dp:
        epsilon, best_alpha = privacy_engine.accountant.get_privacy_spent(
            delta=args.delta
        )
        print(
            f"Train Epoch: {epoch} \t"
            f"Train Loss: {np.mean(losses):.6f} "
            f"Train Accuracy: {np.mean(accuracies):.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}"
        )
    else:
        print(
            f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f} \t Accuracy: {np.mean(accuracies):.6f}"
        )


def evaluate(args, model, test_loader):
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    device = torch.device(args.device)
    model = model.eval()

    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            if i % 10 == 0:
                data = data.to(device)
                label = label.to(device)

                predictions = model(data).squeeze(1)

                loss = criterion(predictions, label)
                acc = binary_accuracy(predictions, label)

                losses.append(loss.item())
                accuracies.append(acc.item())

    mean_accuracy = np.mean(accuracies)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n".format(
            np.mean(losses), mean_accuracy * 100
        )
    )
    return mean_accuracy


def split_train(args, models, train_loader, test_loader, optimizers, privacy_engine, epoch):
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    device = torch.device(args.device)
    client_model, server_model = models
    client_opt, server_opt = optimizers
    for model in models:
        model.train()

    for data, label in (train_loader):
        data = data.to(device)
        label = label.to(device)

        for opt in optimizers:
            opt.zero_grad()

        client_acts = client_model(data)
        server_acts = torch.empty_like(client_acts, requires_grad=True)
        server_acts.data = client_acts.data
        # server_acts.data = client_acts.data / abs(client_acts.data).max()
        # print('max acts check: ', abs(client_acts.data).max().item())

        if args.enable_dp:
            size = server_acts.size()
            noise = Normal(0.0, args.sigma).sample(sample_shape=size)
            server_acts.data.add_(noise.to(server_acts.device))

        server_acts.retain_grad()

        if args.enable_denoise:
            acts = server_acts.data.clone().detach()
            n = args.avg_count
            loss_avg, acc_avg = 0, 0
            for i in range(n):
                drop = nn.Dropout(args.dropout_ratio, inplace=False)
                server_acts.data = drop(acts.data) * (1-args.dropout_ratio)
                server_acts.data.mul_(args.scaling_factor)
                predictions = server_model(server_acts).squeeze(1)
                # if args.enable_denoise:
                #     predictions *= 0.1
                loss = criterion(predictions, label)
                acc = binary_accuracy(predictions, label)
                loss_avg += loss
                acc_avg += acc
                loss.backward()
            loss = loss_avg / n
            acc = acc_avg / n
            server_acts.grad.data.div_(n)
            for name,p in server_model.named_parameters():
                p.grad.data.div_(n)
                # if 'weight' in name:
                #     p.grad.data.mul_(5)


        else:
            predictions = server_model(server_acts).squeeze(1)
            loss = criterion(predictions, label)
            acc = binary_accuracy(predictions, label)
            loss.backward()

        client_acts.grad = server_acts.grad
        client_acts.backward(client_acts.grad)

        # if args.weight_decay:
        #     for model in models:
        #         for name,p in model.named_parameters():
        #             p.grad.data.add_(p.data, alpha=args.weight_decay)

        client_opt.step()
        server_opt.step()

        losses.append(loss.item())
        accuracies.append(acc.item())
        if args.step % 1 == 0:
            print(f"Train Epoch: {epoch} \t Step: {args.step}\t Loss: {np.mean(losses):.6f}\t Accuracy: {np.mean(accuracies):.6f}")
        if args.step % 10 == 0:
            split_evaluate(args, models, test_loader)
            for model in models:
                model.train()
        args.step += 1
    # print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f} \t Accuracy: {np.mean(accuracies):.6f}")


def split_evaluate(args, models, test_loader):
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    device = torch.device(args.device)
    client_model, server_model = models
    for model in models:
        model.eval()

    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            if i % 10 == 0:
                data = data.to(device)
                label = label.to(device)

                predictions = server_model(client_model(data)).squeeze(1)

                loss = criterion(predictions, label)
                acc = binary_accuracy(predictions, label)

                losses.append(loss.item())
                accuracies.append(acc.item())

    mean_accuracy = np.mean(accuracies)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n".format(
            np.mean(losses), mean_accuracy * 100
        )
    )
    args.best_acc = max(args.best_acc, mean_accuracy)
    print(f'Best accuracy: {args.best_acc}')
    return mean_accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Opacus IMDB Example", formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=64, metavar="B", help="input batch size for test", )
    parser.add_argument(
        "-n", "--epochs", type=int, default=10, metavar="N", help="number of epochs to train", )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate", )
    # parser.add_argument(
    #     "--sigma", type=float, default=0.56, metavar="S", help="Noise multiplier", )
    parser.add_argument(
        "-c", "--max-per-sample-grad_norm", type=float, default=1.0, metavar="C", help="Clip per-sample gradients to this norm", )
    parser.add_argument(
        "--delta", type=float, default=1e-5, metavar="D", help="Target delta (default: 1e-5)", )
    parser.add_argument(
        "--max-sequence-length", type=int, default=256, metavar="SL", help="Longer sequences will be cut to this length", )
    parser.add_argument(
        "--device", type=str, default="cuda", help="GPU ID for this process", )
    parser.add_argument(
        "--save-model", action="store_true", default=False, help="Save the trained model", )
    parser.add_argument(
        "--disable-dp", action="store_true", default=False, help="Disable privacy training and just train with vanilla optimizer", )
    parser.add_argument(
        "--secure-rng", action="store_true", default=False, help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost", )
    parser.add_argument(
        "--data-root", type=str, default="../imdb", help="Where IMDB is/will be stored"
    )
    parser.add_argument(
        "-j", "--workers", default=2, type=int, metavar="N", help="number of data loading workers", )
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
    parser.add_argument('--weight-decay', type=float, default=0.0, metavar='M',
                        help='weight decay factor')
    args = parser.parse_args()
    args.step = 0
    device = torch.device(args.device)
    
    #preprocess dataset
    data_root = "../imdb"
    max_sequence_length = 256
    raw_dataset = load_dataset("imdb", cache_dir=data_root)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    dataset = raw_dataset.map(
        lambda x: tokenizer(
            x["text"], truncation=True, max_length=max_sequence_length
        ),
        batched=True,
    )
    dataset.set_format(type="torch", columns=["input_ids", "label"])

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_loader = DataLoader(
        train_dataset,
        num_workers=args.workers,
        batch_size=args.batch_size,
        collate_fn=padded_collate,
        pin_memory=True,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=padded_collate,
        pin_memory=True,
    )

    model = SampleNet(vocab_size=len(tokenizer)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for name, p in model.named_parameters():
        print(name, p.size())

    args.split = True if args.split_layer is not None else False
    args.epochs = 1
    args.dropout_ratio = 1 - args.mask_ratio

    print("All configurations:\n", args)
    if args.enable_dp:
        print(f"dp is enabled! Std of the Gaussian noise: {args.sigma}")
    if args.weight_decay:
        print("weight_decay is enabled! decay parameter:", args.weight_decay)
    if args.enable_denoise:
        if args.scaling_factor != 1.0:
            print(f'scaling is enabled! scaling factor: {args.scaling_factor}')
        if args.dropout_ratio:
            print(f'masking is enabled! ratio: {args.dropout_ratio}, averaging count: {args.avg_count}')
        else:
            args.avg_count = 1

    if args.split:
        client_model = nn.Sequential(*nn.ModuleList(model.children())[:args.split_layer])
        server_model = nn.Sequential(*nn.ModuleList(model.children())[args.split_layer:])
        client_opt = optim.Adam(client_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        server_opt = optim.Adam(server_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model = client_model, server_model
        print(model)
        optimizer = client_opt, server_opt

    privacy_engine = None
    if not args.disable_dp:
        privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)

        # TODO: we need to switch poisson sampling back on, but the
        # model exhibits strange behaviour with batch_size=1
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=args.sigma,
            max_grad_norm=args.max_per_sample_grad_norm,
            poisson_sampling=False,
        )

    mean_accuracy = 0
    args.best_acc = 0
    for epoch in range(1, args.epochs + 1):
        if args.split:
            split_train(args, model, train_loader, test_loader, optimizer, privacy_engine, epoch)
            split_evaluate(args, model, test_loader)
        else:
            train(args, model, train_loader, test_loader, optimizer, privacy_engine, epoch)
            # mean_accuracy = evaluate(args, model, test_loader)

if __name__ == "__main__":
    main()