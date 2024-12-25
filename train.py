import math
import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from os.path import join

from models import SeqGen, ActTrajGen
from data_loaders import get_dataloader, get_dataloader_wo_dist


def train_seqgen(cfg):
    exp_dir = f'ckpt/exp_{cfg["exp_id"]}'
    os.makedirs(exp_dir, exist_ok=True)

    model = SeqGen(cfg).to(cfg['device'])
    optimizer = Adam(params=model.parameters(), lr=cfg['lr'])
    train_loader = get_dataloader(cfg, 'train')
    valid_loader = get_dataloader(cfg, 'valid')

    min_loss = math.inf
    for epoch in range(cfg['epoch_num']):
        model.train()
        train_loss = {'total_loss': 0, 'clf_loss': 0, 'dur_loss': 0, 'dist_loss': 0}
        for itr, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            loss_dict = model.calc_loss(batch)
            loss = loss_dict['total_loss']
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            clip_grad_norm_(model.parameters(), max_norm=cfg['max_norm'], norm_type=cfg['norm_type'])
            optimizer.step()
            for key, val in loss_dict.items():
                train_loss[key] += val
        for key, val in train_loss.items():
            train_loss[key] = val / len(train_loader)

        model.eval()
        valid_loss = {'total_loss': 0, 'clf_loss': 0, 'dur_loss': 0, 'dist_loss': 0}
        for batch in tqdm(valid_loader):
            with torch.no_grad():
                loss_dict = model.calc_loss(batch)
            for key, val in loss_dict.items():
                valid_loss[key] += val.item()
        for key, val in valid_loss.items():
            valid_loss[key] = val / len(valid_loader)
        print(f'Epoch {epoch} \n '
              f'train loss {[f"{key}: {val:.4f}" for key, val in train_loss.items()]} \n'
              f'valid loss {[f"{key}: {val:.4f}" for key, val in valid_loss.items()]}')

        if valid_loss['total_loss'] < min_loss:
            print('Save model ...... ')
            min_loss = valid_loss['total_loss']
            torch.save(model.state_dict(), join(exp_dir, f'seqgen.pth'))


def train_act_gen(cfg):
    exp_dir = f'ckpt/exp_{cfg["exp_id"]}'
    os.makedirs(exp_dir, exist_ok=True)

    model = ActTrajGen(cfg).to(cfg['device'])
    optimizer = Adam(params=model.parameters(), lr=cfg['lr'])
    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.8, patience=5)
    train_loader = get_dataloader_wo_dist(cfg, 'train')
    valid_loader = get_dataloader_wo_dist(cfg, 'valid')

    min_loss = math.inf
    for epoch in range(cfg['epoch_num']):
        model.train()
        train_loss = {'total_loss': 0, 'act_loss': 0, 'time_loss': 0}
        for itr, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            loss_dict = model.calc_loss(batch)
            loss = loss_dict['total_loss']
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            clip_grad_norm_(model.parameters(), max_norm=cfg['max_norm'], norm_type=cfg['norm_type'])
            optimizer.step()
            for key, val in loss_dict.items():
                train_loss[key] += val
        for key, val in train_loss.items():
            train_loss[key] = val / len(train_loader)

        model.eval()
        valid_loss = {'total_loss': 0, 'act_loss': 0, 'time_loss': 0}
        for batch in tqdm(valid_loader):
            with torch.no_grad():
                loss_dict = model.calc_loss(batch)
            for key, val in loss_dict.items():
                valid_loss[key] += val.item()
        for key, val in valid_loss.items():
            valid_loss[key] = val / len(valid_loader)
        cur_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch} lr: {cur_lr: .9f} \n'
              f'train loss {[f"{key}: {val:.4f}" for key, val in train_loss.items()]} \n'
              f'valid loss {[f"{key}: {val:.4f}" for key, val in valid_loss.items()]}')

        if valid_loss['total_loss'] < min_loss:
            print('Save model ...... ')
            min_loss = valid_loss['total_loss']
            torch.save(model.state_dict(), join(exp_dir, f'actgen.pth'))
        lr_scheduler.step(valid_loss['total_loss'])

