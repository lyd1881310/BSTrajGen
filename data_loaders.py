import yaml
import numpy as np
import pandas as pd
import torch
from os.path import join
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class ActDataset(Dataset):
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
        traj_df['weekday'] = traj_df['timestamp'].apply(lambda ts: ts.weekday())
        for _, tdf in traj_df.groupby('traj_id'):
            usr_id = tdf.iloc[0]['usr_id']
            item = {
                'usr_type': usr_type[usr_id],
                'usr_feature': torch.tensor(usr_feature[usr_id], dtype=torch.float32),
                'weekday': tdf.iloc[0]['weekday'],
                'x_act': torch.LongTensor(tdf.iloc[:-1]['act_id'].to_numpy()),
                'y_act': torch.LongTensor(tdf.iloc[1:]['act_id'].to_numpy())
            }
            self.data_list.append(item)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def collator(self, indices):
        usr_type = torch.LongTensor([item['usr_type'] for item in indices])
        usr_feature = torch.stack([item['usr_feature'] for item in indices], dim=0)
        weekday = torch.LongTensor([item['weekday'] for item in indices])
        x_act = pad_sequence([item['x_act'] for item in indices], batch_first=True, padding_value=self.act_pad)
        y_act = pad_sequence([item['y_act'] for item in indices], batch_first=True, padding_value=self.act_pad)
        mask = (x_act == self.act_pad).bool()
        batch = {
            'usr_type': usr_type,
            'usr_feature': usr_feature,
            'weekday': weekday,
            'x_act': x_act,
            'y_act': y_act,
            'mask': mask
        }
        return batch


def get_act_dataloader(cfg, phase):
    data_dir = 'cleared_data'
    name = cfg['dataset']
    batch_size = cfg['batch_size']
    data_dir = join(data_dir, name)

    traj_df = pd.read_csv(join(data_dir, f'{phase}.csv'))
    usr_df = pd.read_csv(join(data_dir, f'usr_feature.csv'))

    dataset = ActDataset(cfg=cfg, usr_df=usr_df, traj_df=traj_df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collator)
    return dataloader

