import os
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from collections import Counter

import yaml

from data_loaders import Scaler


plt.rcParams['font.size'] = 14  # 设置全局字体大小


def plot_day_hour_distri(values):
    bins = np.linspace(0, 24, 25)
    freq, bin_edges = np.histogram(values, bins=bins)
    freq = freq / freq.sum()
    plt.bar(bin_edges[:-1], freq, width=np.diff(bin_edges)[0], edgecolor='black', align='edge')

    plt.title('The distribution of hours in a day')
    plt.xlabel('Hour')
    plt.ylabel('Frequency')
    # 设置x轴刻度
    plt.xticks(np.arange(0, 25, 1), np.arange(0, 25, 1).astype(str))
    plt.xlim([0, 24])  # 确保x轴范围正确
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # 添加网格线


def plot_act_distri(value_counts):
    labels = value_counts.keys()
    values = value_counts.values()
    plt.figure(figsize=(10, 12))
    plt.bar(labels, values, edgecolor='black', alpha=0.7, width=1.0)
    plt.xlabel('Activity category')
    plt.ylabel('Visit frequency')
    plt.xticks(rotation=90)  # 旋转x轴标签，以便更好地显示
    plt.grid(axis='y', linestyle='--')  # 添加y轴网格线


def load_real_df():
    data_dir = 'cleared_data/fsq_global'
    traj_df = pd.read_csv(join(data_dir, 'test.csv'))
    traj_df['timestamp'] = traj_df['timestamp'].apply(lambda ts: pd.Timestamp(ts))
    traj_df['hour_time'] = traj_df['timestamp'].apply(lambda ts: ts.hour + ts.minute / 60)
    return traj_df[['traj_id', 'hour_time', 'category', 'category_id', 'dur', 'dist']]


def load_gen_df(exp_id):
    exp_dir = f'ckpt/exp_{exp_id}'
    data_dir = 'cleared_data/fsq_global'
    scaler = Scaler(dataset='fsq_global')
    gen_df = pd.read_csv(join(exp_dir, 'gen_test.csv'))
    # cate_to_id = json.load(open(join(data_dir, 'category_id.json'), 'r'))
    # id_to_cate = {val: key for key, val in cate_to_id.items()}

    traj_dfs = []
    gen_groups = gen_df.groupby('traj_id')
    for _, group in gen_groups:
        dur = group['time'].diff().iloc[1:].to_numpy()
        group['dur'] = [0] + scaler.denormalize_dur(dur).tolist()
        dist = group['dist'].iloc[1:].to_numpy()
        group['dist'] = [0] + scaler.denormalize_dist(dist).tolist()
        group['hour_time'] = np.cumsum(group['dur']) + group.iloc[0]['time']
        traj_dfs.append(group)
    traj_df = pd.concat(traj_dfs)
    return traj_df[['traj_id', 'hour_time', 'act', 'act_id', 'dur', 'dist']]


def visual_act_traj(exp_id, state):
    data_dir = 'cleared_data/fsq_global'
    visual_dir = f'ckpt/exp_1/visual'
    os.makedirs(visual_dir, exist_ok=True)

    state = 'real'
    if state == 'real':
        traj_df = load_real_df()
    else:
        assert state == 'gen'
        traj_df = load_gen_df()

    # 活动分布
    # cate_to_id = json.load(open(join(data_dir, 'category_id.json'), 'r'))
    cate_to_id = yaml.safe_load(open('cleared_data/activity_id.yaml', 'r'))
    category = [key for key, _ in cate_to_id.items()]
    value_counts = traj_df['act'].value_counts().to_dict()
    value_counts = {cate: value_counts.get(cate, 0) / len(traj_df) for cate in category}
    plt.clf()
    plot_act_distri(value_counts)
    plt.savefig(join(visual_dir, f'{state}_act.pdf'), format='pdf')

    # 时段分布
    plt.clf()
    dur_vals = traj_df[traj_df['dur'] > 0]['dur'].to_numpy()
    sns.kdeplot(dur_vals, fill=True)
    plt.savefig(join(visual_dir, f'{state}_dur.pdf'), format='pdf')

    # 距离分布
    plt.clf()
    dist_vals = traj_df[traj_df['dist'] > 0]['dist'].to_numpy()
    sns.kdeplot(dist_vals, fill=True)
    plt.savefig(join(visual_dir, f'{state}_dist.pdf'), format='pdf')

    # 时间分布
    plt.clf()
    time_vals = traj_df['hour_time'].to_numpy()
    plot_day_hour_distri(time_vals)
    plt.savefig(join(visual_dir, f'{state}_hour.pdf'), format='pdf')


def visual_compare_values():
    """
    真实值和生成值画在一起对比
    """
    visual_dir = 'ckpt/exp_0/visual'
    os.makedirs(visual_dir, exist_ok=True)

    real_df = load_real_df()
    gen_df = load_gen_df()

    plt.clf()
    real_dur = real_df[real_df['dur'] > 0]['dur'].to_numpy()
    gen_dur = gen_df[gen_df['dur'] > 0]['dur'].to_numpy()
    sns.kdeplot(real_dur, label='real', fill=True)
    sns.kdeplot(gen_dur, label='gen', fill=True)
    plt.legend()
    plt.savefig(join(visual_dir, 'dur_cmp.pdf'), format='pdf')

    plt.clf()
    real_dist = real_df[real_df['dist'] > 0]['dist'].to_numpy()
    gen_dist = gen_df[gen_df['dist'] > 0]['dist'].to_numpy()
    sns.kdeplot(real_dist, label='real', fill=True)
    sns.kdeplot(gen_dist, label='gen', fill=True)
    plt.legend()
    plt.savefig(join(visual_dir, 'dist_cmp.pdf'), format='pdf')


def save_gen_trajs(exp_id, mode):
    trg_len = 9
    exp_dir = f'ckpt/exp_{exp_id}'
    os.makedirs(join(exp_dir, 'visual'), exist_ok=True)

    if mode == 'gen':
        traj_df = pd.read_csv(join(exp_dir, 'gen_test.csv'))
    elif mode == 'real':
        traj_df = pd.read_csv('cleared_data/fsq_global/test.csv')
    else:
        raise ValueError
    traj_groups = traj_df.groupby('traj_id')
    hop_df = traj_groups.size().reset_index(name='hop_len')

    trg_ids = hop_df[hop_df['hop_len'] == trg_len]['traj_id'].tolist()
    print(len(trg_ids))
    sample_tid = random.sample(trg_ids, 10)
    print(sample_tid)
    sample_df = pd.concat([traj_groups.get_group(tid) for tid in sample_tid])
    sample_df.to_csv(join(exp_dir, 'visual', f'{mode}_sample_{trg_len}.csv'), index=False)


def visualize_gen_time_distri(exp_id):
    top_k = 7
    exp_dir = f'ckpt/exp_{exp_id}'
    # gen_df = pd.read_csv(join(exp_dir, 'actgen_generate.csv'))
    gen_df = pd.read_csv(join(exp_dir, f'temporal_gen_{top_k}.csv'))
    counts = gen_df.groupby(by=['act', 'time_slot']).size().unstack(fill_value=0)
    counts = counts.div(counts.sum(axis=1), axis=0)
    counts.columns = counts.columns / 2
    # 绘制矩形热度图
    plt.clf()
    plt.figure(figsize=(25, 6))
    sns.heatmap(counts, fmt='d', cmap='YlGnBu')
    plt.title('Act-Hour Heatmap')
    plt.xlabel('Hour')
    plt.ylabel('Activity')
    plt.savefig(join(exp_dir, f'gen_act_time_{top_k}.pdf'), format='pdf')


def visualize_work_act_traj():
    data_dir = 'cleared_data/fsq_global'
    save_dir = 'visualize/work_traj'
    os.makedirs(save_dir, exist_ok=True)

    traj_df = pd.read_csv(join(data_dir, 'traj.csv'))
    usr_groups = traj_df.groupby('usr_id')
    usr_info = []
    for usr_id, group in usr_groups:
        act_freq = {
            act: count / len(group)
            for act, count in group['act'].value_counts().to_dict().items()
        }
        usr_info.append({
            'usr_id': usr_id,
            'office': act_freq.get('office', 0),
            'checkin_num': len(group)
        })
    usr_info_df = pd.DataFrame(usr_info)
    usr_info_df = usr_info_df.sort_values('office', ascending=False).head(20)
    for usr_id in usr_info_df['usr_id'].tolist():
        group = usr_groups.get_group(usr_id)
        group.to_csv(join(save_dir, f'{usr_id}_work_traj.csv'), index=False)


if __name__ == '__main__':
    # visual_act_traj()
    # visual_compare_values()
    # save_gen_trajs(1, 'real')
    visualize_gen_time_distri(exp_id=2)
    # visualize_work_act_traj()
