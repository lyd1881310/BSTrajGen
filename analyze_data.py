import ast
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os.path import join

import yaml
from tqdm import tqdm


def visualize_act_hour_distri():
    data_dir = 'cleared_data/fsq_global'
    visual_dir = join(data_dir, 'visual')
    os.makedirs(visual_dir, exist_ok=True)

    traj_df = pd.read_csv(join(data_dir, 'traj_input.csv'))
    traj_df['timestamp'] = pd.to_datetime(traj_df['timestamp'])
    traj_df['hour'] = traj_df['timestamp'].apply(lambda ts: ts.hour)

    counts = traj_df.groupby(by=['act', 'hour']).size().unstack(fill_value=0)
    counts = counts.div(counts.sum(axis=1), axis=0)
    # 绘制矩形热度图
    plt.figure(figsize=(10, 6))
    sns.heatmap(counts, fmt='d', cmap='YlGnBu')
    plt.title('Act-Hour Heatmap')
    plt.xlabel('Hour')
    plt.ylabel('Activity')
    plt.savefig(join(visual_dir, 'act_hour.pdf'), format='pdf')
    # plt.show()


def visualize_act_time_distri():
    data_dir = 'cleared_data/fsq_global'
    visual_dir = join(data_dir, 'visual')
    os.makedirs(visual_dir, exist_ok=True)

    use_resample = True
    suffix = 'resample' if use_resample else 'ori'
    act_to_id = yaml.safe_load(open('cleared_data/activity_id.yaml', 'r'))
    if use_resample:
        traj_df = pd.read_csv(join(data_dir, 'traj_resample.csv'))
    else:
        traj_df = pd.read_csv(join(data_dir, 'traj.csv'))

    traj_df['is_weekend'] = traj_df['weekday'].apply(lambda day: day > 4)
    tdfs = {'weekend': traj_df[traj_df['is_weekend']], 'weekday': traj_df[~traj_df['is_weekend']]}
    for label, tdf in tdfs.items():
        counts = tdf.groupby(by=['act', 'time_slot']).size().unstack(fill_value=0)
        counts = counts.div(counts.sum(axis=1), axis=0)
        counts.columns = counts.columns / 2
        # 绘制矩形热度图
        plt.clf()
        plt.figure(figsize=(25, 6))
        sns.heatmap(counts, fmt='d', cmap='YlGnBu')
        plt.title('Act-Hour Heatmap')
        plt.xlabel('Hour')
        plt.ylabel('Activity')
        plt.savefig(join(visual_dir, f'{label}_act_time_{suffix}.pdf'), format='pdf')

        act_count = tdf['act'].value_counts().to_dict()
        acts = [act for act, _ in act_to_id.items()]
        freq = [act_count.get(act, 0) / len(tdf) for act in acts]
        plt.clf()
        plt.figure(figsize=(7, 10))
        plt.bar(acts, freq)
        plt.xticks(rotation=90)
        plt.xlabel('Act')
        plt.ylabel('Freq')
        plt.savefig(join(visual_dir, f'{label}_act_freq_{suffix}.pdf'), format='pdf')


if __name__ == '__main__':
    # describe_libcity_instagram()
    # describe_libcity_fsq()
    # describe_sand_fsq()
    # visualize_fsq_nyc()
    # visualize_act_hour_distri()
    visualize_act_time_distri()