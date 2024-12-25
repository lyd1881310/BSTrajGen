import json
import os
import ast
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


def prepare_fsq_nyc_poi():
    ori_dir = 'ori_data/libcity_fsq_nyc'
    data_dir = 'cleared_data/fsq_nyc'
    os.makedirs(data_dir, exist_ok=True)

    # 处理 POI
    poi_df = pd.read_csv(join(ori_dir, 'foursquare_nyc.geo'))
    poi_mapping = [
        (['home', 'residential', 'apartment'], 'residential'),
        (['food', 'restaurant', 'deli', 'bodega', 'pizza', 'fried', 'chicken', 'bakery', 'bbq joint'], 'food'),
        (['nightlife', 'entertainment', 'art', 'music', 'spar'], 'entertainment'),
        (['medical', 'drugstore', 'pharmacy'], 'medical'),


        (['office'], 'office'),
        (['outdoors', 'recreation'], 'outdoor'),
        # ['Professional & Other Places']: '',
        (['college', 'university', 'library', 'school'], 'education'),
        (['gym', 'athletics', 'sport'], 'sports'),
        (['theater'], 'theater'),
        (['shop', 'service'], 'service'),

        (['hotel'], 'hotel'),
        (['bar', 'caffe', 'tea'], 'bar'),
        (['station', 'subway', 'road', 'airport'], 'transportation'),
        (['park', 'garden', 'playground'], 'park'),
        (['bank'], 'bank'),
        (['store', 'market', 'mall'], 'store'),
        (['building'], 'building')
    ]

    def convert(venue_str):
        for keys, act in poi_mapping:
            if any([key in venue_str.lower() for key in keys]):
                return act
        return 'other'
        # return venue_str.lower()
    poi_df['category'] = poi_df['venue_category_name'].apply(lambda venue: convert(venue))
    name_to_id = {name: idx for idx, (_, name) in enumerate(poi_mapping + [(None, 'other')])}
    json.dump(name_to_id, open(join(data_dir, 'category_id.json'), 'w'), indent=4)
    poi_df['category_id'] = poi_df['category'].apply(lambda c: name_to_id[c])
    category_cnt = poi_df['category'].value_counts()
    print(category_cnt)

    poi_df['lon'] = poi_df['coordinates'].apply(lambda crds: ast.literal_eval(crds)[0])
    poi_df['lat'] = poi_df['coordinates'].apply(lambda crds: ast.literal_eval(crds)[1])
    poi_df = poi_df[['geo_id', 'category', 'category_id', 'lon', 'lat']].rename(columns={'geo_id': 'poi_id'})
    poi_df.to_csv(join(data_dir, 'poi.csv'), index=False)


def prepare_fsq_nyc_traj():
    ori_dir = 'ori_data/libcity_fsq_nyc'
    data_dir = 'cleared_data/fsq_nyc'

    poi_df = pd.read_csv(join(data_dir, 'poi.csv'))
    traj_df = pd.read_csv(join(ori_dir, 'foursquare_nyc.dyna'))
    traj_df['timestamp'] = traj_df['time'].apply(lambda ts: pd.Timestamp(ts))
    poi_df = poi_df.set_index('poi_id', drop=False)
    usr_groups = traj_df.groupby('entity_id')

    utm_epsg = calc_utm_epsg(poi_df['lon'].mean(), poi_df['lat'].mean())
    latlon2xy = pyproj.Transformer.from_crs(4326, utm_epsg)

    traj_id = 0
    traj_dfs = []
    for usr_id, group in tqdm(usr_groups, total=len(usr_groups)):
        group = group.sort_values('timestamp')

        # 根据时间间隔切分同一个用户的轨迹
        group['dur'] = group['timestamp'].diff().fillna(pd.Timedelta(0)).apply(lambda delta: delta.total_seconds() / 60)
        group['ref_tid'] = np.cumsum(group['dur'] > 24 * 60)

        for _, tdf in group.groupby('ref_tid'):
            tdf['dur'] = tdf['dur'].shift(-1).fillna(0)
            tdf['lon'] = tdf['location'].apply(lambda loc: poi_df.loc[loc, 'lon'])
            tdf['lat'] = tdf['location'].apply(lambda loc: poi_df.loc[loc, 'lat'])
            tdf['category'] = tdf['location'].apply(lambda loc: poi_df.loc[loc, 'category'])
            tdf['category_id'] = tdf['location'].apply(lambda loc: poi_df.loc[loc, 'category_id'])

            tdf['x'], tdf['y'] = latlon2xy.transform(tdf['lat'].to_numpy(), tdf['lon'].to_numpy())
            tdf['dx'] = (tdf['x'].shift(-1) - tdf['x']).fillna(0)
            tdf['dy'] = (tdf['y'].shift(-1) - tdf['y']).fillna(0)
            tdf['dist'] = tdf.apply(lambda row: np.sqrt(row['dx'] ** 2 + row['dy'] ** 2), axis=1)
            tdf['is_explore'] = 1
            for row_num, (index, row) in enumerate(tdf.iterrows()):
                history = tdf.iloc[:row_num]['location'].tolist()
                if row['location'] in history:
                    tdf.loc[index, 'is_explore'] = 0
            tdf['traj_id'] = traj_id
            traj_id += 1
            tdf = tdf[[
                'entity_id', 'traj_id', 'timestamp', 'location', 'lon', 'lat',
                'category', 'category_id', 'dur', 'dist', 'is_explore'
            ]].rename(columns={'entity_id': 'usr_id'})
            if len(tdf) < 3:
                continue
            traj_dfs.append(tdf)
    traj_df = pd.concat(traj_dfs)
    print(f'Check-in num {len(traj_df)} User num {traj_df["usr_id"].nunique()} Traj num {traj_df["traj_id"].nunique()}')
    traj_df.to_csv(join(data_dir, 'traj.csv'), index=False)


def split_train_data(dataset):
    data_dir = 'cleared_data'
    # traj_df = pd.read_csv(join(data_dir, dataset, 'traj.csv'))
    # traj_df = pd.read_csv(join(data_dir, dataset, 'traj_filter.csv'))
    traj_df = pd.read_csv(join(data_dir, dataset, 'traj_input.csv'))
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


def pre_process_fsq_global():
    ori_dir = 'ori_data/dataset_TIST2015'
    data_dir = 'cleared_data/fsq_global'
    os.makedirs(data_dir, exist_ok=True)

    city_df = pd.read_csv(join(ori_dir, 'dataset_TIST2015_Cities.txt'), header=None, sep='\t',
                          names=['city_name', 'lat', 'lon', 'country_code', 'country_name', 'city_type'])
    print(city_df.head().to_string())
    print(city_df['country_name'].value_counts().iloc[:10])

    # print(poi_df['venue_category_name'].value_counts().iloc[:40])
    #
    # category_count = poi_df['venue_category_name'].value_counts().reset_index()
    # category_count.columns = ['category_name', 'count']
    # category_count['freq'] = category_count['count'] / category_count['count'].sum()
    # category_count['cum_freq'] = np.cumsum(category_count['freq'])
    # category_count = category_count.round(decimals={'freq': 4, 'cum_freq': 4})
    # category_count.to_csv(join(data_dir, 'ori_category.csv'), index=False)


def prepare_fsq_global_poi():
    ori_dir = 'ori_data/dataset_TIST2015'
    data_dir = 'cleared_data/fsq_global'

    print('prepare global poi ...... ')
    mapping = yaml.safe_load(open(join(data_dir, 'category_mapping.yaml'), 'r'))
    category_to_idx = {category: idx for idx, category in enumerate(mapping)}
    json.dump(category_to_idx, open(join(data_dir, 'category_id.json'), 'w'), indent=4)

    def get_mapping_category(ori_cate):
        for category, ori_list in mapping.items():
            if any([ori.lower() in ori_cate.lower() for ori in ori_list]):
                return category
        return 'other'

    poi_df = pd.read_csv(
        join(ori_dir, 'dataset_TIST2015_POIs.txt'), header=None, sep='\t',
        names=['venue_id', 'lat', 'lon', 'ori_category', 'country_code']
    )
    poi_df['category'] = poi_df['ori_category'].apply(get_mapping_category)
    poi_df['category_id'] = poi_df['category'].apply(lambda cate: category_to_idx[cate])
    poi_df['poi_id'] = range(len(poi_df))
    poi_df = poi_df[['poi_id', 'venue_id', 'ori_category', 'category', 'category_id', 'country_code', 'lon', 'lat']]
    poi_df.to_csv(join(data_dir, 'poi.csv'), index=False)


def calc_distance(tdf: pd.DataFrame):
    prev_lon, prev_lat = tdf.iloc[:-1]['lon'].tolist(), tdf.iloc[:-1]['lat'].tolist()
    cur_lon, cur_lat = tdf.iloc[1:]['lon'].tolist(), tdf.iloc[1:]['lat'].tolist()
    dist = [0] + [
        geodesic((lat0, lon0), (lat1, lon1)).kilometers
        for lon0, lat0, lon1, lat1 in zip(prev_lon, prev_lat, cur_lon, cur_lat)
    ]
    return dist


def calc_dur(timestamp: pd.Series):
    deltas = timestamp.diff().fillna(pd.Timedelta(0))
    dur_list = deltas.apply(lambda delta: delta.total_seconds() / 3600).tolist()
    return dur_list


def get_usr_traj_dfs(usr_data):
    usr_group = usr_data['usr_group']
    poi_df = usr_data['poi_df']

    min_hops = 3
    max_hop_dist = 20  # km
    max_hop_dur = 24  # hour

    usr_group['timestamp'] = usr_group.apply(
        lambda row: pd.Timestamp(row['utc_time']) + pd.Timedelta(row['tz_offset'], unit='m'), axis=1
    )
    usr_group['category'] = usr_group['venue_id'].apply(lambda vid: poi_df.loc[vid, 'category'])
    usr_group['category_id'] = usr_group['venue_id'].apply(lambda vid: poi_df.loc[vid, 'category_id'])
    usr_group['lon'] = usr_group['venue_id'].apply(lambda vid: poi_df.loc[vid, 'lon'])
    usr_group['lat'] = usr_group['venue_id'].apply(lambda vid: poi_df.loc[vid, 'lat'])

    usr_group = usr_group.sort_values('timestamp')
    # 时间单位 hour
    usr_group['dur'] = calc_dur(usr_group['timestamp'])
    usr_group['dist'] = calc_distance(usr_group)
    usr_group['ref'] = np.cumsum((usr_group['dur'] > max_hop_dur) | (usr_group['dist'] > max_hop_dist))

    traj_dfs = []
    sub_groups = usr_group.groupby('ref')
    for _, sub_group in tqdm(sub_groups, total=len(sub_groups)):
        # sub_group = drop_duplicate(sub_group)
        if len(sub_group) < min_hops:
            continue
        sub_group['dist'] = sub_group['dist'].shift(-1).fillna(0)
        sub_group['dur'] = sub_group['dur'].shift(-1).fillna(0)
        traj_dfs.append(sub_group)
    return traj_dfs


def prepare_fsq_global_traj():
    print('prepare global traj ...... ')
    ori_dir = 'ori_data/dataset_TIST2015'
    data_dir = 'cleared_data/fsq_global'

    t0 = time.time()

    poi_df = pd.read_csv(join(data_dir, 'poi.csv'))
    poi_df = poi_df.set_index('venue_id', drop=False)
    traj_df = pd.read_csv(
        join(ori_dir, 'dataset_TIST2015_Checkins.txt'), header=None, sep='\t',
        names=['usr_id', 'venue_id', 'utc_time', 'tz_offset'],
        # nrows=1000000
    )

    top_usr_num = 10000
    usr_checkin_num = traj_df['usr_id'].value_counts().head(top_usr_num).reset_index()
    usr_checkin_num.columns = ['usr_id', 'count']
    usr_checkin_num.to_csv(join(data_dir, 'usr_checkin_num.csv'), index=False)

    t1 = time.time()
    print(f'Count time {t1 - t0} sec')

    usr_groups = traj_df.groupby('usr_id')
    usr_groups = [
        {
            'usr_group': usr_groups.get_group(row['usr_id']),
            'poi_df': poi_df
        }
        for _, row in usr_checkin_num.iterrows()
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
    traj_df = traj_df[['usr_id', 'traj_id', 'timestamp', 'category', 'category_id', 'dur', 'dist', 'lon', 'lat']]
    traj_df.to_csv(join(data_dir, 'traj_new.csv'), index=False)


def prepare_day_traj(dataset):
    data_dir = 'cleared_data'
    traj_df = pd.read_csv(join(data_dir, dataset, 'traj.csv'))

    # 删除 other
    traj_df = traj_df[traj_df['category'] != 'other']

    traj_df['timestamp'] = traj_df['timestamp'].apply(lambda ts: pd.Timestamp(ts))
    traj_df['date'] = traj_df['timestamp'].apply(lambda ts: ts.date())
    traj_df['usr_date'] = traj_df.apply(lambda row: f'{row["usr_id"]}-{row["traj_id"]}-{row["date"]}', axis=1)

    # 删除高频记录
    filter_dfs =[]
    min_hop, max_hop = 4, 16
    min_dist, max_dist = 0, 20
    traj_groups = traj_df.groupby('usr_date')
    for _, group in tqdm(traj_groups, total=len(traj_groups)):
        group['dur'] = calc_dur(group['timestamp'])
        drop = (group['dur'] * 60 < 15) | (group['category_id'] == group['category_id'].shift())
        group = group[~drop].copy()
        if len(group) < min_hop or len(group) > max_hop:
            continue
        group['dist'] = calc_distance(group)
        if group['dist'].max() > max_dist:
            continue
        group['dur'] = calc_dur(group['timestamp'])
        filter_dfs.append(group)
    filter_df = pd.concat(filter_dfs)
    filter_df['traj_id'] = np.cumsum(filter_df['usr_date'] != filter_df['usr_date'].shift())
    del filter_df['date']
    del filter_df['usr_date']
    filter_df.to_csv(join(data_dir, dataset, 'traj_filter.csv'), index=False)


def pre_process_fsq_local(dataset):
    ori_dir = 'ori_data/dataset_tsmc2014'
    # 注意原始数据文件字符编码
    if dataset == 'fsq_nyc':
        traj_df = pd.read_csv(
            join(ori_dir, 'dataset_TSMC2014_NYC.txt'), header=None, sep='\t', encoding='ISO-8859-1',
            names=['usr_id', 'venue_id', 'venue_category_id', 'venue_category_name', 'lat', 'lon', 'tz_offset', 'utc_time']
        )
    elif dataset == 'fsq_tky':
        traj_df = pd.read_csv(
            join(ori_dir, 'dataset_TSMC2014_TKY.txt'), header=None, sep='\t', encoding='ISO-8859-1',
            names=['usr_id', 'venue_id', 'venue_category_id', 'venue_category_name', 'lat', 'lon', 'tz_offset', 'utc_time']
        )
    else:
        raise ValueError

    data_dir = join('cleared_data', dataset)
    os.makedirs(data_dir, exist_ok=True)

    category_count = traj_df['venue_category_name'].value_counts().reset_index()
    category_count.columns = ['category_name', 'count']
    category_count['freq'] = category_count['count'] / category_count['count'].sum()
    category_count['cum_freq'] = np.cumsum(category_count['freq'])
    category_count = category_count.round(decimals={'freq': 4, 'cum_freq': 4})
    category_count.to_csv(join(data_dir, 'ori_category.csv'), index=False)


def merge_activity(dataset='fsq_global'):
    """
    活动类型进一步聚焦到基本的十种
    """
    act_to_id = yaml.safe_load(open('cleared_data/activity_id.yaml', 'r'))
    data_dir = join('cleared_data', dataset)
    act_mapping = yaml.safe_load(open(join(data_dir, 'activity.yaml'), 'r'))
    cate_to_act = {
        cate: act
        for act, cate_list in act_mapping.items()
        for cate in cate_list
    }
    print(cate_to_act)
    traj_df = pd.read_csv(join(data_dir, 'traj_filter.csv'))

    traj_df['act'] = traj_df['category'].apply(lambda cate: cate_to_act[cate])
    traj_df['act_id'] = traj_df['act'].apply(lambda act: act_to_id[act])
    traj_df = traj_df[['usr_id', 'traj_id', 'timestamp', 'act', 'act_id', 'dur', 'dist', 'lon', 'lat']]
    traj_df.to_csv(join(data_dir, 'traj_input.csv'), index=False)


if __name__ == '__main__':
    # prepare_fsq_nyc_poi()
    # prepare_fsq_nyc_traj()
    # prepare_fsq_nyc_train()

    # pre_process_fsq_local('fsq_nyc')
    # pre_process_fsq_local('fsq_tky')
    # pre_process_fsq_global()

    # prepare_fsq_global_poi()

    # prepare_fsq_global_traj()
    # prepare_day_traj('fsq_global')

    merge_activity('fsq_global')
    split_train_data('fsq_global')
