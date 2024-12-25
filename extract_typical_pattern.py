import pandas as pd
from os.path import join


def get_typical_traj():
    data_dir = 'cleared_data/fsq_global'
    traj_df = pd.read_csv(join(data_dir, 'traj_input.csv'))
    usr_groups = traj_df.groupby('usr_id')
    typical_users = []
    for usr_id, group in usr_groups:
        total_count = len(group)
        act_count = group['act'].value_counts().to_dict()
        residential = act_count.get('residential', 0)
        office = act_count.get('office', 0)
        if residential / total_count > 0.2 and office / total_count > 0.2:
            typical_users.append(usr_id)
    print(len(typical_users))
    for usr_id in typical_users:
        usr_df = usr_groups.get_group(usr_id)
        usr_df.to_csv(join(data_dir, 'visual', f'usr_{usr_id}.csv'), index=False)


if __name__ == '__main__':
    get_typical_traj()
