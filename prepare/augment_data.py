import numpy as np
import pandas as pd
from os.path import join
from tqdm import tqdm
import yaml


def get_act_ratio(traj_df: pd.DataFrame):
    act_counts = traj_df['act'].value_counts().reset_index()
    act_counts.columns = ['act', 'count']
    act_counts['ratio'] = act_counts['count'] / act_counts['count'].sum()
    return act_counts


def resample_act_traj():
    """
    根据不同活动的权重重新采样, 得到训练用的数据集
    """
    data_dir = 'cleared_data/fsq_global'
    act_weight = yaml.safe_load(open(join(data_dir, 'sample_weight.yaml'), 'r'))
    traj_df = pd.read_csv(join(data_dir, 'traj.csv'))

    ori_act_ratio = get_act_ratio(traj_df)
    print(ori_act_ratio.to_string())

    # 重新采样得到的轨迹数
    resample_num = traj_df['traj_id'].nunique()

    # 计算采样权重
    traj_scores = dict()
    traj_groups = traj_df.groupby('traj_id')
    for traj_id, group in tqdm(traj_groups, total=len(traj_groups)):
        act_count = group['act'].value_counts().to_dict()
        score = 0
        for act, wt in act_weight.items():
            score += act_count.get(act, 0) / len(group) * wt
        traj_scores[traj_id] = score

    traj_ids = list(traj_scores.keys())
    scores = np.array(list(traj_scores.values()))
    prob = scores / scores.sum()
    sample_ids = np.random.choice(a=len(prob), p=prob, size=resample_num)

    resample_dfs = []
    for new_tid, idx in tqdm(enumerate(sample_ids), total=len(sample_ids)):
        tid = traj_ids[idx]
        tdf = traj_groups.get_group(tid).copy()
        tdf['traj_id'] = new_tid
        resample_dfs.append(tdf)
    resample_df = pd.concat(resample_dfs)
    resample_df.to_csv(join(data_dir, 'resample_traj.csv'))

    act_ratio = get_act_ratio(resample_df)
    print(act_ratio.to_string())


if __name__ == '__main__':
    resample_act_traj()
