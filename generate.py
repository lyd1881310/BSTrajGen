import numpy as np
import pandas as pd
import torch
import yaml
import random
from os.path import join
from tqdm import tqdm
from argparse import ArgumentParser

from act_generator import ActTrajGen
from time_generator import TemporalGenerator


def add_residential_act(acts, trg_act_id, prob=0.98):
    """
    活动序列后处理:
        + 在每天轨迹的开始和结尾添加居住型活动
        + 按概率执行此操作
    """
    if acts[0] != trg_act_id and random.random() < prob:
        acts = [trg_act_id] + acts
    if acts[-1] != trg_act_id and random.random() < prob:
        acts.append(trg_act_id)
    return acts


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
        weekday = group.iloc[0]['weekday']
        batch = {
            'seq_len': len(group),
            'usr_type': torch.LongTensor([usr_type]),
            'usr_feature': torch.tensor(usr_feature, dtype=torch.float32).unsqueeze(0),
            'weekday': torch.LongTensor([weekday]),
            'x_act': torch.LongTensor([start_act]).unsqueeze(0)
        }
        gen_batch = model.generate(batch)
        acts = gen_batch['act'].squeeze().cpu().tolist()

        # 后处理操作
        acts = add_residential_act(acts, act_to_id['residential'])

        gen_list += [
            {'traj_id': traj_id, 'weekday': weekday, 'act': id_to_act[act_id], 'act_id': act_id}
            for act_id in acts
        ]
        traj_id += 1

    gen_df = pd.DataFrame(gen_list)
    gen_df.to_csv(join(exp_dir, 'act_gen.csv'), index=False)


def run_time_generate(exp_id):
    top_k = 7
    data_dir = 'cleared_data/fsq_global'
    exp_dir = f'ckpt/exp_{exp_id}'
    gen_df = pd.read_csv(join(exp_dir, 'act_gen.csv'))
    weekday_distri = np.load(join(data_dir, 'weekday_distri.npy'))
    weekend_distri = np.load(join(data_dir, 'weekend_distri.npy'))
    generator = TemporalGenerator(weekday_distri=weekday_distri, weekend_distri=weekend_distri, top_k=top_k)

    gen_dfs = []
    traj_groups = gen_df.groupby('traj_id')
    for traj_id, group in tqdm(traj_groups, total=len(traj_groups)):
        tdf = group.copy()
        is_weekend = tdf.iloc[0]['weekday'] > 4
        act_list = tdf['act_id'].tolist()
        if len(act_list) > 48:
            continue
        tdf['time_slot'] = generator.generate(act_list=act_list, is_weekend=is_weekend)
        gen_dfs.append(tdf)
    gen_df = pd.concat(gen_dfs)
    gen_df['hour'] = gen_df['time_slot'].apply(lambda slot: (30 * slot + random.randint(0, 29)) / 60)
    gen_df.to_csv(join(exp_dir, f'time_gen_{top_k}.csv'), index=False)

