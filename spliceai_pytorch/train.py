import argparse
import tqdm
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from spliceai_pytorch import SpliceAI_80nt
from sklearn.metrics import average_precision_score

import wandb
import os
import numpy as np

def shuffle(arr):
    return np.random.choice(arr, size=len(arr), replace=False)

def top_k_accuracy(pred_probs, labels):
    pred_probs, labels = map(lambda x: x.view(-1), [pred_probs, labels])  # Flatten
    k = (labels == 1.0).sum().item()

    _, top_k_indices = pred_probs.topk(k)
    correct = labels[top_k_indices] == 1.0
    return correct.float().mean()

def train(model, h5f, train_shard_idxs, batch_size, optimizer, criterion):
    model.train()
    running_output, running_label = [], []

    batch_idx = 0
    for i, shard_idx in enumerate(shuffle(train_shard_idxs), 1):
        X = h5f[f'X{shard_idx}'][:].transpose(0, 2, 1)
        Y = h5f[f'Y{shard_idx}'][0, ...]

        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)  # TODO: Check whether drop_last=True?
        
        bar = tqdm.tqdm(loader, leave=False, total=len(loader), desc=f'Shard {i}/{len(train_shard_idxs)}')
        for batch in bar:
            X, Y = batch[0].cuda(), batch[1].cuda()
            optimizer.zero_grad()
            out = model(X) # (batch_size, 5000, 3)
            loss = criterion(out.reshape(-1, 3), Y.argmax(dim=-1).view(-1))
            loss.backward()
            optimizer.step()

            running_output.append(out.detach().cpu())
            running_label.append(Y.detach().cpu())

            if batch_idx % 1000 == 0:
                running_output = torch.cat(running_output, dim=0)
                running_label = torch.cat(running_label, dim=0)
                running_output_prob = running_output.softmax(dim=-1)

                top_k_acc_1 = top_k_accuracy(running_output_prob[:, :, 1], running_label[:, :, 1])
                top_k_acc_2 = top_k_accuracy(running_output_prob[:, :, 2], running_label[:, :, 2])
                auprc_1 = average_precision_score(running_label[:, :, 1].view(-1), running_output_prob[:, :, 1].view(-1))
                auprc_2 = average_precision_score(running_label[:, :, 2].view(-1), running_output_prob[:, :, 2].view(-1))

                loss = criterion(running_output.reshape(-1, 3), running_label.argmax(dim=-1).view(-1))
                bar.set_postfix(loss=f'{loss.item():.4f}', topk_acc=f'{top_k_acc_1.item():.4f}', topk_don=f'{top_k_acc_2.item():.4f}', auprc_acc=f'{auprc_1:.4f}', auprc_don=f'{auprc_2:.4f}')

                running_output, running_label = [], []

                wandb.log({
                    'train/loss': loss.item(),
                    'train/topk_acceptor': top_k_acc_1.item(),
                    'train/topk_donor': top_k_acc_2.item(),
                    'train/auprc_acceptor': auprc_1,
                    'train/auprc_donor': auprc_2,
                })
            
            batch_idx += 1


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

    out = torch.cat(out, dim=0)
    label = torch.cat(label, dim=0)
    out_prob = out.softmax(dim=-1)

    loss = criterion(out.reshape(-1, 3), label.argmax(dim=-1).view(-1))
    top_k_acc_1 = top_k_accuracy(out_prob[:, :, 1], label[:, :, 1])
    top_k_acc_2 = top_k_accuracy(out_prob[:, :, 2], label[:, :, 2])
    auprc_1 = average_precision_score(label[:, :, 1].view(-1), out_prob[:, :, 1].view(-1))
    auprc_2 = average_precision_score(label[:, :, 2].view(-1), out_prob[:, :, 2].view(-1))
    
    print(f'Val loss: {loss.item():.4f}, topk_acc: {top_k_acc_1.item():.4f}, topk_don: {top_k_acc_2.item():.4f}, auprc_acc: {auprc_1:.4f}, auprc_don: {auprc_2:.4f}')

    wandb.log({
        'val/loss': loss.item(),
        'val/topk_acceptor': top_k_acc_1.item(),
        'val/topk_donor': top_k_acc_2.item(),
        'val/auprc_acceptor': auprc_1,
        'val/auprc_donor': auprc_2,
    })

    return loss.item()

def test(model, h5f, test_shard_idxs, batch_size, criterion):
    model.eval()

    out, label = [], []
    for shard_idx in test_shard_idxs:
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

    out = torch.cat(out, dim=0)
    label = torch.cat(label, dim=0)
    out_prob = out.softmax(dim=-1)

    loss = criterion(out.reshape(-1, 3), label.argmax(dim=-1).view(-1))
    top_k_acc_1 = top_k_accuracy(out_prob[:, :, 1], label[:, :, 1])
    top_k_acc_2 = top_k_accuracy(out_prob[:, :, 2], label[:, :, 2])
    auprc_1 = average_precision_score(label[:, :, 1].view(-1), out_prob[:, :, 1].view(-1))
    auprc_2 = average_precision_score(label[:, :, 2].view(-1), out_prob[:, :, 2].view(-1))
    
    print(f'Test loss: {loss.item():.4f}, topk_acc: {top_k_acc_1.item():.4f}, topk_don: {top_k_acc_2.item():.4f}, auprc_acc: {auprc_1:.4f}, auprc_don: {auprc_2:.4f}')

    wandb.log({
        'test/loss': loss.item(),
        'test/topk_acceptor': top_k_acc_1.item(),
        'test/topk_donor': top_k_acc_2.item(),
        'test/auprc_acceptor': auprc_1,
        'test/auprc_donor': auprc_2,
    })

    return loss.item()

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
    import h5py
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-h5', required=True)
    parser.add_argument('--test-h5', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', '-b', type=int, default=18)
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-wandb', action='store_true', default=False)
    args = parser.parse_args()

    if not args.use_wandb:
        os.environ['WANDB_MODE'] = 'disabled'

    wandb.init(project='spliceai-pytorch', config=args, reinit=True)
    seed_everything(args.seed)

    train_h5f = h5py.File(args.train_h5, 'r')
    test_h5f = h5py.File(args.test_h5, 'r')

    num_shards = len(train_h5f.keys()) // 2
    shard_idxs = np.random.permutation(num_shards)
    train_shard_idxs = shard_idxs[:int(0.9 * num_shards)]
    val_shard_idxs = shard_idxs[int(0.9 * num_shards):]

    test_shard_idxs = np.arange(len(test_h5f.keys()) // 2)

    model = SpliceAI_80nt()
    model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6, 7, 8, 9], gamma=0.5)

    for epoch in range(args.epochs):
        train(model, train_h5f, train_shard_idxs, args.batch_size, optimizer, criterion)
        validate(model, train_h5f, val_shard_idxs, args.batch_size, criterion)
        test(model, test_h5f, test_shard_idxs, args.batch_size, criterion)

        scheduler.step()

if __name__ == '__main__':
    main()