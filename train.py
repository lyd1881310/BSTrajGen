import math
import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from os.path import join

from act_generator import ActTrajGen
from data_loaders import get_act_dataloader


def train_act_gen(cfg):
    exp_dir = f'ckpt/exp_{cfg["exp_id"]}'
    os.makedirs(exp_dir, exist_ok=True)

    model = ActTrajGen(cfg).to(cfg['device'])
    optimizer = Adam(params=model.parameters(), lr=cfg['lr'])
    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.8, patience=5)
    train_loader = get_act_dataloader(cfg, 'train')
    valid_loader = get_act_dataloader(cfg, 'valid')

    min_loss = math.inf
    for epoch in range(cfg['epoch_num']):
        model.train()
        train_loss = 0
        for itr, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            loss = model.calc_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            clip_grad_norm_(model.parameters(), max_norm=cfg['max_norm'], norm_type=cfg['norm_type'])
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        valid_loss = 0
        for batch in tqdm(valid_loader):
            with torch.no_grad():
                loss = model.calc_loss(batch)
            valid_loss += loss.item()
        valid_loss /= len(valid_loader)

        cur_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch} lr: {cur_lr: .9f} train loss {train_loss:.4f} valid loss {valid_loss: .4f}')

        if valid_loss < min_loss:
            print('Save model ...... ')
            min_loss = valid_loss
            torch.save(model.state_dict(), join(exp_dir, f'actgen.pth'))
        lr_scheduler.step(valid_loss)

