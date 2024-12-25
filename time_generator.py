import random

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

