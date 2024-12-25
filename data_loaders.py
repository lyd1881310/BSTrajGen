import yaml
import numpy as np
import pandas as pd
import torch
from os.path import join
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class ActTrajDataset(Dataset):
    def __init__(self, cfg, usr_df, traj_df):
        self.event_pad = cfg['category_num']
        self.dur_pad = 0.
        self.dist_pad = 0.
        self.hour_pad = 0
        self.weekday_pad = 0

        self.data_list = []
        usr_feature_cols = [col for col in usr_df.columns if col != 'usr_id' and col != 'label']
        usr_feature = {
            row['usr_id']: np.array(row[usr_feature_cols], dtype=np.float32)
            for _, row in usr_df.iterrows()
        }
        usr_type = {row['usr_id']: row['label'] for _, row in usr_df.iterrows()}

        traj_df['timestamp'] = traj_df['timestamp'].apply(lambda ts: pd.Timestamp(ts))
        traj_df['hour'] = traj_df['timestamp'].apply(lambda ts: ts.hour)
        traj_df['weekday'] = traj_df['timestamp'].apply(lambda ts: ts.weekday())
        for _, tdf in traj_df.groupby('traj_id'):
            usr_id = tdf.iloc[0]['usr_id']
            item = {
                'usr_type': usr_type[usr_id],
                'usr_feature': torch.tensor(usr_feature[usr_id], dtype=torch.float32),
                'x_event': torch.LongTensor(tdf.iloc[:-1]['act_id'].to_numpy()),
                'hour': torch.LongTensor(tdf.iloc[:-1]['hour'].to_numpy()),
                'weekday': torch.LongTensor(tdf.iloc[:-1]['weekday'].to_numpy()),
                'y_event': torch.LongTensor(tdf.iloc[1:]['act_id'].to_numpy()),
                'dur': torch.tensor(tdf.iloc[1:]['dur'].to_numpy(np.float32), dtype=torch.float32),
                'dist': torch.tensor(tdf.iloc[1:]['dist'].to_numpy(np.float32), dtype=torch.float32),
            }
            self.data_list.append(item)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def collator(self, indices):
        usr_type = torch.LongTensor([item['usr_type'] for item in indices])
        usr_feature = torch.stack([item['usr_feature'] for item in indices], dim=0)
        x_event = pad_sequence([item['x_event'] for item in indices], batch_first=True, padding_value=self.event_pad)
        hour = pad_sequence([item['hour'] for item in indices], batch_first=True, padding_value=self.hour_pad)
        weekday = pad_sequence([item['weekday'] for item in indices], batch_first=True, padding_value=self.weekday_pad)

        y_event = pad_sequence([item['y_event'] for item in indices], batch_first=True, padding_value=self.event_pad)
        dur = pad_sequence([item['dur'] for item in indices], batch_first=True, padding_value=self.dur_pad)
        dist = pad_sequence([item['dist'] for item in indices], batch_first=True, padding_value=self.dist_pad)
        # mask = torch.tensor(x_event == self.event_pad, dtype=torch.float32)
        mask = (x_event == self.event_pad).bool()
        batch = {
            'usr_type': usr_type,
            'usr_feature': usr_feature,
            'x_event': x_event,
            'hour': hour,
            'weekday': weekday,
            'y_event': y_event,
            'dur': dur,
            'dist': dist,
            'mask': mask
        }
        return batch


class ActDatasetWoDist(Dataset):
    def __init__(self, cfg, usr_df, traj_df):
        self.act_pad = cfg['act_num']
        self.time_pad = cfg['time_slot_num']
        self.weekday_pad = 7

        self.data_list = []
        usr_feature_cols = [col for col in usr_df.columns if col != 'usr_id' and col != 'label']
        usr_feature = {
            row['usr_id']: np.array(row[usr_feature_cols], dtype=np.float32)
            for _, row in usr_df.iterrows()
        }
        usr_type = {row['usr_id']: row['label'] for _, row in usr_df.iterrows()}

        traj_df['timestamp'] = traj_df['timestamp'].apply(lambda ts: pd.Timestamp(ts))
        traj_df['time_slot'] = traj_df['timestamp'].apply(lambda ts: ts.hour)
        traj_df['weekday'] = traj_df['timestamp'].apply(lambda ts: ts.weekday())
        for _, tdf in traj_df.groupby('traj_id'):
            usr_id = tdf.iloc[0]['usr_id']
            item = {
                'usr_type': usr_type[usr_id],
                'usr_feature': torch.tensor(usr_feature[usr_id], dtype=torch.float32),
                'x_act': torch.LongTensor(tdf.iloc[:-1]['act_id'].to_numpy()),
                'x_time': torch.LongTensor(tdf.iloc[:-1]['time_slot'].to_numpy()),
                'weekday': tdf.iloc[0]['weekday'],
                'y_act': torch.LongTensor(tdf.iloc[1:]['act_id'].to_numpy()),
                'y_time': torch.LongTensor(tdf.iloc[1:]['time_slot'].to_numpy())
            }
            self.data_list.append(item)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def collator(self, indices):
        usr_type = torch.LongTensor([item['usr_type'] for item in indices])
        usr_feature = torch.stack([item['usr_feature'] for item in indices], dim=0)
        x_act = pad_sequence([item['x_act'] for item in indices], batch_first=True, padding_value=self.act_pad)
        x_time = pad_sequence([item['x_time'] for item in indices], batch_first=True, padding_value=self.time_pad)
        # weekday 每条轨迹对应一个 weekday, 不需要 pad
        weekday = torch.LongTensor([item['weekday'] for item in indices])
        # weekday = pad_sequence([item['weekday'] for item in indices], batch_first=True, padding_value=self.weekday_pad)
        y_act = pad_sequence([item['y_act'] for item in indices], batch_first=True, padding_value=self.act_pad)
        y_time = pad_sequence([item['y_time'] for item in indices], batch_first=True, padding_value=self.time_pad)
        mask = (x_act == self.act_pad).bool()
        batch = {
            'usr_type': usr_type,
            'usr_feature': usr_feature,
            'x_act': x_act,
            'x_time': x_time,
            'weekday': weekday,
            'y_act': y_act,
            'y_time': y_time,
            'mask': mask
        }
        return batch


class Scaler:
    def __init__(self, dataset):
        data_dir = join('cleared_data', dataset)
        # traj_df = pd.read_csv(join(data_dir, 'traj_filter.csv'))
        traj_df = pd.read_csv(join(data_dir, 'traj_input.csv'))
        dur_vals = traj_df[traj_df['dur'] > 0]['dur'].to_numpy()
        dist_vals = traj_df[traj_df['dist'] > 0]['dist'].to_numpy()
        self.min_dur = dur_vals.min()
        self.max_dur = dur_vals.max()
        self.min_dist = dist_vals.min()
        self.max_dist = dist_vals.max()

    def transform(self, traj_df):
        traj_df['dur'] = (traj_df['dur'] - self.min_dur) / (self.max_dur - self.min_dur)
        traj_df['dist'] = (traj_df['dist'] - self.min_dist) / (self.max_dist - self.min_dist)
        return traj_df

    def normalize_dur(self, dur_data):
        return (dur_data - self.min_dur) / (self.max_dur - self.min_dur)

    def normalize_dist(self, dist_data):
        return (dist_data - self.min_dist) / (self.max_dist - self.min_dist)

    def denormalize_dur(self, dur_data):
        return self.min_dur + dur_data * (self.max_dur - self.min_dur)

    def denormalize_dist(self, dist_data):
        return self.min_dist + dist_data * (self.max_dist - self.min_dist)


def get_dataloader(cfg, phase):
    data_dir = 'cleared_data'
    name = cfg['dataset']
    batch_size = cfg['batch_size']
    data_dir = join(data_dir, name)

    traj_df = pd.read_csv(join(data_dir, f'{phase}.csv'))
    usr_df = pd.read_csv(join(data_dir, f'usr_feature.csv'))

    scaler = Scaler(name)
    traj_df = scaler.transform(traj_df)

    dataset = ActTrajDataset(cfg=cfg, usr_df=usr_df, traj_df=traj_df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collator)
    return dataloader


def get_dataloader_wo_dist(cfg, phase):
    data_dir = 'cleared_data'
    name = cfg['dataset']
    batch_size = cfg['batch_size']
    data_dir = join(data_dir, name)

    traj_df = pd.read_csv(join(data_dir, f'{phase}.csv'))
    usr_df = pd.read_csv(join(data_dir, f'usr_feature.csv'))

    dataset = ActDatasetWoDist(cfg=cfg, usr_df=usr_df, traj_df=traj_df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collator)
    return dataloader
