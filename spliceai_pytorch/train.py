import argparse
import tqdm
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from spliceai_pytorch import SpliceAI_80nt
import numpy as np

def shuffle(arr):
    return np.random.choice(arr, size=len(arr), replace=False)

def train(model, h5f, train_shard_idxs, batch_size, optimizer, criterion):
    model.train()
    running_output, running_label = [], []

    for i, shard_idx in enumerate(shuffle(train_shard_idxs), 1):
        X = h5f[f'X{shard_idx}'][:].transpose(0, 2, 1)
        Y = h5f[f'Y{shard_idx}'][0, ...]

        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        
        bar = tqdm.tqdm(loader, leave=False, total=len(loader), desc=f'Shard {i}/{len(train_shard_idxs)}')
        for idx, batch in enumerate(bar):
            X, Y = batch[0].cuda(), batch[1].cuda()
            optimizer.zero_grad()
            out = model(X) # (batch_size, 5000, 3)
            loss = criterion(out, Y)
            loss.backward()
            optimizer.step()

            running_output.append(out.detach().cpu())
            running_label.append(Y.detach().cpu())

            if idx % 100 == 0:
                running_output = torch.cat(running_output, dim=0)
                running_label = torch.cat(running_label, dim=0)

                loss = criterion(running_output, running_label)
                bar.set_postfix(loss=f'{loss.item():.4f}')

                running_output, running_label = [], []


def validate(model, h5f, val_shard_idxs, batch_size, criterion):
    model.eval()

    out, label = [], []
    for shard_idx in val_shard_idxs:
        X = h5f[f'X{shard_idx}'][:].transpose(0, 2, 1)
        Y = h5f[f'Y{shard_idx}'][0, ...]

        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        bar = tqdm.tqdm(loader, leave=False, total=len(loader))
        for idx, batch in enumerate(bar):
            X, Y = batch[0].cuda(), batch[1].cuda()
            _out = model(X).detach().cpu()
            _label = Y.detach().cpu()

            out.append(_out)
            label.append(_label)

    loss = criterion(torch.cat(out, dim=0), torch.cat(label, dim=0))
    return loss.item()

def test(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_accuracy

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Performance drops, so commenting out for now.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

def main():
    import pandas as pd
    import h5py
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-h5', required=True)
    parser.add_argument('--test-h5', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', '-b', type=int, default=6)
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    train_h5f = h5py.File(args.train_h5, 'r')
    test_h5f = h5py.File(args.test_h5, 'r')

    num_shards = len(train_h5f.keys()) // 2
    shard_idxs = np.random.permutation(num_shards)
    train_shard_idxs = shard_idxs[:int(0.9 * num_shards)]
    val_shard_idxs = shard_idxs[int(0.9 * num_shards):]

    model = SpliceAI_80nt()
    model.cuda()

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6, 7, 8, 9], gamma=0.5)

    for epoch in range(args.epochs):
        train(model, train_h5f, train_shard_idxs, args.batch_size, optimizer, criterion)
        validate(model, train_h5f, val_shard_idxs, args.batch_size, criterion)


        scheduler.step()


if __name__ == '__main__':
    main()