import numpy as np
import pandas as pd
import torch
import yaml
from os.path import join
from tqdm import tqdm
from argparse import ArgumentParser

from models import SeqGen, ActTrajGen


def run_generate(cfg):
    exp_id = cfg['exp_id']
    data_dir = join('cleared_data', cfg['dataset'])
    exp_dir = f'ckpt/exp_{exp_id}'

    act_to_id = yaml.safe_load(open('cleared_data/activity_id.yaml', 'r'))
    id_to_act = {act_id: act for act, act_id in act_to_id.items()}

    model = SeqGen(cfg).to(cfg['device'])
    model.load_state_dict(torch.load(join(exp_dir, 'seqgen.pth'), map_location=cfg['device'], weights_only=True))

    test_df = pd.read_csv(join(data_dir, 'test.csv'))
    test_df['timestamp'] = test_df['timestamp'].apply(lambda ts: pd.Timestamp(ts))
    test_df['hour_time'] = test_df['timestamp'].apply(lambda ts: ts.hour + ts.minute / 60)
    usr_df = pd.read_csv(join(data_dir, 'usr_feature.csv'))
    usr_cols = [col for col in usr_df.columns if col != 'usr_id' and col != 'label']
    usr_id_to_type = {row['usr_id']: row['label'] for _, row in usr_df.iterrows()}
    usr_id_to_feature = {row['usr_id']: row[usr_cols].to_numpy(dtype=np.float32) for _, row in usr_df.iterrows()}

    test_groups = test_df.groupby('traj_id')
    gen_list = []
    traj_id = 0
    # 逐条生成, batch_size = 1
    for _, group in tqdm(test_groups, total=len(test_groups)):
        usr_id = group.iloc[0]['usr_id']
        usr_type = int(usr_id_to_type[usr_id])
        usr_feature = usr_id_to_feature[usr_id]
        start_event = group.iloc[0]['act_id']
        start_time = group.iloc[0]['hour_time']
        start_hour = group.iloc[0]['timestamp'].hour
        start_weekday = group.iloc[0]['timestamp'].weekday()
        batch = {
            'seq_len': len(group),
            'time': torch.tensor(start_time, dtype=torch.float32).unsqueeze(0),
            'usr_type': torch.LongTensor([usr_type]),
            'usr_feature': torch.tensor(usr_feature, dtype=torch.float32).unsqueeze(0),
            'x_event': torch.LongTensor([start_event]).unsqueeze(0),
            'hour': torch.LongTensor([start_hour]).unsqueeze(0),
            'weekday': torch.LongTensor([start_weekday]).unsqueeze(0)
        }
        gen_batch = model.generate(batch)

        events = gen_batch['event'].cpu().tolist()
        times = gen_batch['time'].cpu().tolist()
        dists = gen_batch['dist'].cpu().tolist()
        for seq_event, seq_time, seq_dist in zip(events, times, dists):
            gen_list += [
                {'traj_id': traj_id, 'time': t, 'act': id_to_act[e], 'act_id': e, 'dist': d}
                for e, t, d in zip(seq_event, seq_time, seq_dist)
            ]
            traj_id += 1
    gen_df = pd.DataFrame(gen_list)
    gen_df.to_csv(join(exp_dir, 'gen_test.csv'), index=False)


def run_act_generate(cfg):
    exp_id = cfg['exp_id']
    data_dir = join('cleared_data', cfg['dataset'])
    exp_dir = f'ckpt/exp_{exp_id}'

    act_to_id = yaml.safe_load(open('cleared_data/activity_id.yaml', 'r'))
    id_to_act = {act_id: act for act, act_id in act_to_id.items()}

    model = ActTrajGen(cfg).to(cfg['device'])
    model.load_state_dict(torch.load(join(exp_dir, 'actgen.pth'), map_location=cfg['device'], weights_only=True))

    test_df = pd.read_csv(join(data_dir, 'test.csv'))
    usr_df = pd.read_csv(join(data_dir, 'usr_feature.csv'))
    usr_cols = [col for col in usr_df.columns if col != 'usr_id' and col != 'label']
    usr_id_to_type = {row['usr_id']: row['label'] for _, row in usr_df.iterrows()}
    usr_id_to_feature = {row['usr_id']: row[usr_cols].to_numpy(dtype=np.float32) for _, row in usr_df.iterrows()}

    test_groups = test_df.groupby('traj_id')
    gen_list = []
    traj_id = 0
    # 逐条生成, batch_size = 1
    for _, group in tqdm(test_groups, total=len(test_groups)):
        usr_id = group.iloc[0]['usr_id']
        usr_type = int(usr_id_to_type[usr_id])
        usr_feature = usr_id_to_feature[usr_id]
        start_act = group.iloc[0]['act_id']
        start_time = group.iloc[0]['time_slot']
        weekday = group.iloc[0]['weekday']
        batch = {
            'seq_len': len(group),
            'usr_type': torch.LongTensor([usr_type]),
            'usr_feature': torch.tensor(usr_feature, dtype=torch.float32).unsqueeze(0),
            'x_time': torch.LongTensor([start_time]).unsqueeze(0),
            'x_act': torch.LongTensor([start_act]).unsqueeze(0),
            'weekday': torch.LongTensor([weekday])
        }
        gen_batch = model.generate(batch)
        acts = gen_batch['act'].squeeze().cpu().tolist()
        times = gen_batch['time'].squeeze().cpu().tolist()

        gen_list += [
            {'traj_id': traj_id, 'time_slot': time_slot, 'act': id_to_act[act_id], 'act_id': act_id}
            for act_id, time_slot in zip(acts, times)
        ]
        traj_id += 1

    gen_df = pd.DataFrame(gen_list)
    gen_df.to_csv(join(exp_dir, 'actgen_generate.csv'), index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--exp_id', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='fsq_global')
    parser.add_argument('--device', type=str)
    args = parser.parse_args()

    config = yaml.safe_load(open('config.yaml', 'r'))
    config.update(vars(args))

    run_generate(config)
