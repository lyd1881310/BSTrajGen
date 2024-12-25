import os
import pandas as pd
import numpy as np
from os.path import join

import yaml
from tqdm import tqdm


def get_return_prob():
    """
    计算 checkin 数据中的返回-探索概率 (空间规律)
    """
    data_dir = 'cleared_data/fsq_global'
    traj_df = pd.read_csv(join(data_dir, 'resample_traj.csv'))
    act_to_id = yaml.safe_load(open('cleared_data/activity_id.yaml', 'r'))

    def get_return_act(tdf: pd.DataFrame):
        loc_set = set()
        tdf['is_return'] = False
        for idx, row in tdf.iterrows():
            if row['venue_id'] in loc_set:
                tdf.loc[idx, 'is_return'] = True
            loc_set.add(row['venue_id'])
        return tdf

    usr_dfs = []
    usr_groups = traj_df.groupby('usr_id')
    for usr_id, usr_group in tqdm(usr_groups, total=len(usr_groups)):
        usr_group = get_return_act(usr_group)
        usr_dfs.append(usr_group)
    traj_df = pd.concat(usr_dfs)

    act_groups = traj_df.groupby('act')
    act_return_prob = {
        act: np.sum(group['is_return']) / len(group)
        for act, group in act_groups
    }

    # 约束条件: 关注驻留倾向高的活动
    act_return_prob['residential'] = max(act_return_prob['residential'], 0.98)
    act_return_prob['office'] = max(act_return_prob['office'], 0.92)
    act_return_prob['education'] = max(act_return_prob['education'], 0.95)

    return_prob = [
        {
            'act': act,
            'act_id': act_to_id[act],
            'return_prob': prob
        }
        for act, prob in act_return_prob.items()
    ]
    pd.DataFrame(return_prob).to_csv(join(data_dir, 'return_prob.csv'), index=False)


def get_time_slot_distri():
    """
    计算活动的时间分布规律
    """
    data_dir = 'cleared_data/fsq_global'
    traj_df = pd.read_csv(join(data_dir, 'resample_traj.csv'))
    traj_df['is_weekend'] = traj_df['weekday'].apply(lambda day: day > 4)

    weekday_df = traj_df[~traj_df['is_weekend']]
    weekend_df = traj_df[traj_df['is_weekend']]

    def get_time_distri(tdf):
        counts = tdf.groupby(by=['act_id', 'time_slot']).size().unstack(fill_value=0)
        counts = counts.div(counts.sum(axis=1), axis=0)
        return counts.to_numpy()

    weekday_distri = get_time_distri(weekday_df)
    weekend_distri = get_time_distri(weekend_df)
    np.save(join(data_dir, 'weekday_distri.npy'), weekday_distri)
    np.save(join(data_dir, 'weekend_distri.npy'), weekend_distri)


if __name__ == '__main__':
    get_return_prob()
    get_time_slot_distri()
