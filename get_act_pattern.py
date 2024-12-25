import os
import pandas as pd
import numpy as np
from os.path import join

import yaml
from tqdm import tqdm


def get_return_prob():
    data_dir = 'cleared_data/fsq_global'
    traj_df = pd.read_csv(join(data_dir, 'traj_resample.csv'))
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
    return_prob = [
        {
            'act': act,
            'act_id': act_to_id[act],
            'return_prob': np.sum(group['is_return']) / len(group)
        }
        for act, group in act_groups
    ]
    pd.DataFrame(return_prob).to_csv(join(data_dir, 'return_prob.csv'), index=False)


if __name__ == '__main__':
    get_return_prob()
