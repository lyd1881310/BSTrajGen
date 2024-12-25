import json
import numpy as np
import pandas as pd

from scipy.stats import entropy
from typing import List
from collections import Counter
from os.path import join

from data_loaders import Scaler


def calc_jsd(p: np.ndarray, q: np.ndarray):
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def calc_act_jsd(real_act: List, gen_act: List, act_num: int):
    def get_freq(act_list):
        freq = np.zeros(act_num)
        count = dict(Counter(act_list))
        for key, val in count.items():
            freq[key] = val
        freq = freq / freq.sum()
        return freq
    real_freq = get_freq(real_act)
    gen_freq = get_freq(gen_act)
    return calc_jsd(real_freq, gen_freq)


def calc_value_jsd(real_vals: np.ndarray, gen_vals: np.ndarray, bin_num=30):
    low = min(real_vals.min(), gen_vals.min())
    high = max(real_vals.max(), gen_vals.max())
    edges = np.linspace(low, high, bin_num + 1)
    real_count, _ = np.histogram(real_vals, bins=edges)
    gen_count, _ = np.histogram(gen_vals, bins=edges)
    real_freq = real_count / real_count.sum()
    gen_freq = gen_count / gen_count.sum()
    return calc_jsd(real_freq, gen_freq)


def calc_time_slot_jsd(real_time, gen_time, time_slot_num):
    def get_slot_freq(time_list):
        freq = np.zeros(time_slot_num)
        for time_slot, count in dict(Counter(time_list)).items():
            freq[time_slot] = count
        return freq / freq.sum()

    real_freq = get_slot_freq(real_time)
    gen_freq = get_slot_freq(gen_time)
    return calc_jsd(real_freq, gen_freq)


def run_evaluate(exp_id):
    dataset = 'fsq_global'
    data_dir = 'cleared_data'
    exp_dir = f'ckpt/exp_{exp_id}'
    real_df = pd.read_csv(join(data_dir, dataset, 'test.csv'))
    gen_df = pd.read_csv(join(exp_dir, 'gen_test.csv'))

    scaler = Scaler(dataset)

    real_act = real_df['act_id'].tolist()
    real_dur = real_df[real_df['dur'] > 0]['dur'].to_numpy()
    real_dist = real_df[real_df['dist'] > 0]['dist'].to_numpy()

    gen_act = gen_df['act_id'].tolist()
    gen_dist = gen_df[gen_df['dist'] > 0]['dist'].to_numpy()
    gen_dur = []
    for _, group in gen_df.groupby('traj_id'):
        gen_dur += group['time'].diff().iloc[1:].tolist()
    gen_dur = np.array(gen_dur, dtype=np.float32)

    # 归一化
    real_dur = scaler.normalize_dur(real_dur)
    real_dist = scaler.normalize_dist(real_dist)
    gen_dur = scaler.normalize_dur(gen_dur)
    gen_dist = scaler.normalize_dist(gen_dist)

    act_num = 10
    eval_dict = {
        'act_jsd': calc_act_jsd(real_act=real_act, gen_act=gen_act, act_num=act_num),
        'dur_jsd': calc_value_jsd(real_vals=real_dur, gen_vals=gen_dur),
        'dist_jsd': calc_value_jsd(real_vals=real_dist, gen_vals=gen_dist)
    }
    print(eval_dict)
    json.dump(eval_dict, open(join(exp_dir, 'eval.json'), 'w'), indent=4)


def run_evaluate_act_generate(exp_id):
    data_dir = 'cleared_data/fsq_global'
    exp_dir = f'ckpt/exp_{exp_id}'
    real_df = pd.read_csv(join(data_dir, 'test.csv'))
    # gen_df = pd.read_csv(join(exp_dir, 'actgen_generate.csv'))
    gen_df = pd.read_csv(join(exp_dir, 'temporal_gen_7.csv'))

    real_act = real_df['act_id'].tolist()
    gen_act = gen_df['act_id'].tolist()
    act_jsd = calc_act_jsd(real_act=real_act, gen_act=gen_act, act_num=10)

    real_time = real_df['time_slot'].tolist()
    # gen_time = gen_df['time_slot'].tolist()
    gen_time = gen_df[gen_df['time_slot'] != gen_df['time_slot'].shift()]['time_slot'].tolist()
    time_jsd = calc_time_slot_jsd(real_time=real_time, gen_time=gen_time, time_slot_num=48)

    print(act_jsd, time_jsd)


if __name__ == '__main__':
    # run_evaluate(exp_id=1)
    run_evaluate_act_generate(exp_id=2)
