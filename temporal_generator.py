import pandas as pd
import numpy as np
from os.path import join
from tqdm import tqdm


class TemporalGenerator:
    def __init__(self, weekday_distri: np.ndarray, weekend_distri: np.ndarray, top_k):
        self.weekday_distri = weekday_distri
        self.weekend_distri = weekend_distri
        self.time_num = weekday_distri.shape[1]
        self.top_k = top_k

    def time_slot_sample(self, score, min_t, max_t):
        """
        返回闭区间 [min_t, max_t] 中的一个时间片, 按 score 概率采样
        """
        assert len(score) == self.time_num and min_t <= max_t
        c_score = score[min_t:max_t+1]
        top_indexes = np.argsort(c_score)[-self.top_k:]
        time_to_score = {
            idx + min_t: c_score[idx]
            for idx in top_indexes
        }
        indices = list(time_to_score.keys())
        prob = np.array(list(time_to_score.values()))
        prob = prob / prob.sum()
        idx = np.random.choice(len(prob), p=prob)
        return indices[idx]

    def generate(self, act_list, is_weekend: bool):
        time_distri = self.weekend_distri if is_weekend else self.weekday_distri
        traj_len = len(act_list)
        score = np.zeros((traj_len, self.time_num))
        prev = np.full((traj_len, self.time_num), fill_value=-1)

        for i, act in enumerate(act_list):
            if i == 0:
                score[i] = time_distri[act]
                continue
            for j in range(self.time_num):
                if j < i:
                    continue
                prev_idx = np.argmax(score[i - 1, :j])
                score[i, j] = score[i - 1, prev_idx] + time_distri[act][j]
                prev[i, j] = prev_idx

        time_list = []
        max_t = self.time_num - 1
        for i in range(traj_len - 1, -1, -1):
            t = self.time_slot_sample(score=score[i], min_t=i, max_t=max_t)
            time_list.append(t)
            max_t = t - 1
        time_list = list(reversed(time_list))
        return time_list


def prepare_time_distri():
    data_dir = 'cleared_data/fsq_global'
    traj_df = pd.read_csv(join(data_dir, 'traj.csv'))
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


def run_time_generate():
    top_k = 7
    data_dir = 'cleared_data/fsq_global'
    exp_dir = 'ckpt/exp_2'
    gen_df = pd.read_csv(join(exp_dir, 'actgen_generate.csv'))
    weekday_distri = np.load(join(data_dir, 'weekday_distri.npy'))
    weekend_distri = np.load(join(data_dir, 'weekend_distri.npy'))
    generator = TemporalGenerator(weekday_distri=weekday_distri, weekend_distri=weekend_distri, top_k=top_k)

    gen_dfs = []
    traj_groups = gen_df.groupby('traj_id')
    for traj_id, group in tqdm(traj_groups, total=len(traj_groups)):
        tdf = group.copy()
        # is_weekend = tdf.iloc[0]['weekday'] > 4
        is_weekend = False
        act_list = tdf['act_id'].tolist()
        if len(act_list) > 48:
            continue
        tdf['time_slot'] = generator.generate(act_list=act_list, is_weekend=is_weekend)
        gen_dfs.append(tdf)
    gen_df = pd.concat(gen_dfs)
    gen_df.to_csv(join(exp_dir, f'temporal_gen_{top_k}.csv'), index=False)


if __name__ == '__main__':
    # prepare_time_distri()
    run_time_generate()

