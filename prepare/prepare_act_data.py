import json
import os
import ast
import math
import random
import time
import pyproj
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import yaml
from tqdm import tqdm
from geopy.distance import geodesic
from concurrent.futures import ProcessPoolExecutor


def calc_utm_epsg(lon, lat):
    """
    根据给定的纬度和经度计算适合的 UTM EPSG 编码
    """
    # 计算 UTM 带号（每6度一个带）
    utm_band = int((lon + 180) // 6) + 1
    # 判断是北半球还是南半球，北半球是 326xx，南半球是 327xx
    if lat >= 0:
        epsg_code = f"326{utm_band:02d}"  # 北半球
    else:
        epsg_code = f"327{utm_band:02d}"  # 南半球
    return int(epsg_code)


def resample_act_traj(dataset):
    top_percent = 0.2
    repeat_num = 3
    data_dir = 'cleared_data'
    traj_df = pd.read_csv(join(data_dir, dataset, 'traj.csv'))
    usr_groups = traj_df.groupby('usr_id')
    usr_info = []
    for usr_id, group in usr_groups:
        act_freq = {
            act: count / len(group)
            for act, count in group['act'].value_counts().to_dict().items()
        }
        usr_info.append({
            'usr_id': usr_id,
            'residential': act_freq.get('residential', 0),
            'office': act_freq.get('office', 0),
            'checkin_num': len(group)
        })
    usr_info_df = pd.DataFrame(usr_info)
    usr_info_df['sum'] = usr_info_df['residential'] + usr_info_df['office']
    usr_info_df = usr_info_df.sort_values(by=['sum'], ascending=False)
    top_usr_ids = usr_info_df.head(n=int(len(usr_info_df) * top_percent))['usr_id'].tolist()

    tid = 0
    traj_dfs = []
    for ori_tid, tdf in traj_df.groupby('traj_id'):
        usr_id = tdf.iloc[0]['usr_id']
        traj_count = repeat_num if usr_id in top_usr_ids else 1
        for _ in range(traj_count):
            tdf['traj_id'] = tid
            traj_dfs.append(tdf.copy())
            tid += 1
    resample_df = pd.concat(traj_dfs)
    resample_df.to_csv(join(data_dir, dataset, 'traj_resample.csv'), index=False)


def split_train_data(dataset):
    data_dir = 'cleared_data'
    traj_df = pd.read_csv(join(data_dir, dataset, 'traj_resample.csv'))
    total_ids = traj_df['traj_id'].tolist()
    random.shuffle(total_ids)

    train_size = int(len(total_ids) * 0.7)
    valid_size = int(len(total_ids) * 0.15)

    sampled_ids = {
        'train': total_ids[:train_size],
        'valid': total_ids[train_size: train_size + valid_size],
        'test': total_ids[train_size + valid_size:]
    }
    traj_groups = traj_df.groupby('traj_id')
    for phase, ids in sampled_ids.items():
        phase_df = pd.concat([traj_groups.get_group(tid) for tid in ids])
        phase_df.to_csv(join(data_dir, dataset, f'{phase}.csv'), index=False)


def prepare_fsq_global_poi():
    print('prepare fsq_global poi ...... ')
    ori_dir = 'ori_data/dataset_TIST2015'
    data_dir = 'cleared_data/fsq_global'
    os.makedirs(data_dir, exist_ok=True)

    act_to_category = yaml.safe_load(open(join(data_dir, 'act_to_category.yaml'), 'r'))
    act_to_id = yaml.safe_load(open('cleared_data/activity_id.yaml', 'r'))

    def get_mapping_act(ori_cate):
        for act, ori_list in act_to_category.items():
            if any([ori.lower() in ori_cate.lower() for ori in ori_list]):
                return act
        return 'other'

    poi_df = pd.read_csv(
        join(ori_dir, 'dataset_TIST2015_POIs.txt'), header=None, sep='\t',
        names=['venue_id', 'lat', 'lon', 'ori_category', 'country_code']
    )
    poi_df['act'] = poi_df['ori_category'].apply(get_mapping_act)
    poi_df['act_id'] = poi_df['act'].apply(lambda act: act_to_id[act])

    poi_df['poi_id'] = range(len(poi_df))
    poi_df = poi_df[['poi_id', 'venue_id', 'ori_category', 'act', 'act_id', 'country_code', 'lon', 'lat']]
    poi_df.to_csv(join(data_dir, 'poi.csv'), index=False)


def get_usr_traj_dfs(usr_data):
    usr_traj_df = usr_data['usr_group']
    poi_df = usr_data['poi_df']

    # 数据处理配置
    minute_delta = 30  # 每个时间片的长度
    min_hops, max_hops = 4, 16

    usr_traj_df['timestamp'] = usr_traj_df.apply(
        lambda row: pd.Timestamp(row['utc_time']) + pd.Timedelta(row['tz_offset'], unit='m'), axis=1
    )
    usr_traj_df['date'] = usr_traj_df['timestamp'].apply(lambda ts: ts.date())
    usr_traj_df['weekday'] = usr_traj_df['timestamp'].apply(lambda ts: ts.weekday())
    usr_traj_df['time_slot'] = (usr_traj_df['timestamp']
                                .apply(lambda ts: math.floor((60 * ts.hour + ts.minute) / minute_delta)))

    usr_traj_df['act'] = usr_traj_df['venue_id'].apply(lambda vid: poi_df.loc[vid, 'act'])
    usr_traj_df['act_id'] = usr_traj_df['venue_id'].apply(lambda vid: poi_df.loc[vid, 'act_id'])
    usr_traj_df['lon'] = usr_traj_df['venue_id'].apply(lambda vid: poi_df.loc[vid, 'lon'])
    usr_traj_df['lat'] = usr_traj_df['venue_id'].apply(lambda vid: poi_df.loc[vid, 'lat'])
    usr_traj_df = usr_traj_df.sort_values('timestamp')

    def process_day_traj(tdf: pd.DataFrame):
        mask = (tdf['time_slot'] == tdf['time_slot'].shift())
        tdf = tdf[~mask]
        mask = (tdf['act_id'] == tdf['act_id'].shift())
        tdf = tdf[~mask]
        return tdf

    traj_dfs = []
    groups = usr_traj_df.groupby('date')
    for _, group in groups:
        traj_df = process_day_traj(group)
        if len(traj_df) < min_hops or len(traj_df) > max_hops:
            continue
        traj_dfs.append(traj_df)
    return traj_dfs


def prepare_fsq_global():
    print('prepare global traj ...... ')
    ori_dir = 'ori_data/dataset_TIST2015'
    data_dir = 'cleared_data/fsq_global'
    os.makedirs(data_dir, exist_ok=True)

    t0 = time.time()
    poi_df = pd.read_csv(join(data_dir, 'poi.csv'))
    poi_df = poi_df.set_index('venue_id', drop=False)
    traj_df = pd.read_csv(
        join(ori_dir, 'dataset_TIST2015_Checkins.txt'), header=None, sep='\t',
        names=['usr_id', 'venue_id', 'utc_time', 'tz_offset'],
        # nrows=100000
    )
    top_usr_num = 10000
    usr_checkin_num = traj_df['usr_id'].value_counts().head(top_usr_num).to_dict()
    t1 = time.time()
    print(f'Read time {t1 - t0} sec')

    usr_groups = traj_df.groupby('usr_id')
    usr_groups = [
        {
            'usr_group': usr_groups.get_group(usr_id),
            'poi_df': poi_df
        }
        for usr_id, _ in usr_checkin_num.items()
    ]

    with ProcessPoolExecutor() as executor:
        total_dfs = list(executor.map(get_usr_traj_dfs, usr_groups))

    flatten_dfs = []
    traj_id = 0
    for usr_dfs in total_dfs:
        for tdf in usr_dfs:
            tdf['traj_id'] = traj_id
            flatten_dfs.append(tdf)
            traj_id += 1
    traj_df = pd.concat(flatten_dfs)
    traj_df = traj_df[
        ['usr_id', 'traj_id', 'timestamp', 'weekday', 'time_slot', 'act', 'act_id', 'venue_id', 'lon', 'lat']
    ]
    traj_df.to_csv(join(data_dir, 'traj.csv'), index=False)


if __name__ == '__main__':
    # prepare_fsq_global_poi()
    # prepare_fsq_global()
    resample_act_traj('fsq_global')
    split_train_data('fsq_global')
