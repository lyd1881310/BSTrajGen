import math
import pandas as pd
import numpy as np
from os.path import join
from tqdm import tqdm


def get_act_dur():
    data_dir = 'cleared_data/fsq_global'
    traj_df = pd.read_csv(join(data_dir, 'traj_input.csv'))
    traj_df = traj_df[traj_df['dur'] > 0]
    traj_df['dur_slots'] = traj_df['dur'].apply(lambda dur: math.floor(np.clip(dur / 0.5, 0, 47)))

    groups = traj_df.groupby(by=['act_id', 'dur_slots'])
    act_dur = np.zeros((10, 48))
    for (act_id, dur_slot), group in tqdm(groups, total=len(groups)):
        act_dur[act_id, dur_slot] = len(group)
    print(act_dur)
    np.save(join(data_dir, 'act_dur.npy'), act_dur)


if __name__ == '__main__':
    get_act_dur()
