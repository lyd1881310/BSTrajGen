import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import join
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def get_semantic_feature(traj_df):
    # 构建用户文档
    def strip_word(cate_word):
        return cate_word.replace('&', '').replace(' ', '')

    user_docs = (traj_df.groupby('usr_id')['act']
                 .apply(lambda cate_list: ' '.join([f'{strip_word(c)}' for c in cate_list])).reset_index())
    user_docs.columns = ['usr_id', 'document']

    # 使用 TfidfVectorizer 计算用户画像的 TF-IDF 特征
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(user_docs['document'])
    features = vectorizer.get_feature_names_out()

    # 将 TF-IDF 矩阵转换为 DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=features)
    tfidf_df['usr_id'] = user_docs['usr_id']

    tfidf_df = tfidf_df.sort_values('usr_id')
    return tfidf_df


def get_temporal_feature(traj_df):
    traj_df['timestamp'] = traj_df['timestamp'].apply(lambda ts: pd.Timestamp(ts))

    # TODO: 细粒度时间特征抽取
    time_slot_num = 24
    traj_df['time_slot'] = traj_df['timestamp'].apply(lambda ts: ts.hour)

    usr_ids = []
    time_feature = []
    for usr_id, group in traj_df.groupby('usr_id'):
        value_count = group['time_slot'].value_counts().to_dict()
        usr_time_feature = np.zeros(time_slot_num)
        for key, val in value_count.items():
            usr_time_feature[key] = val
        usr_time_feature = usr_time_feature / usr_time_feature.sum()
        time_feature.append(usr_time_feature)
        usr_ids.append(usr_id)

    usr_time_df = pd.DataFrame(data=np.array(time_feature), columns=[f't{t}' for t in range(time_slot_num)])
    usr_time_df['usr_id'] = usr_ids
    usr_time_df = usr_time_df.sort_values('usr_id')
    return usr_time_df


def extract_usr_pref(dataset):
    data_dir = 'cleared_data'
    traj_df = pd.read_csv(join(data_dir, dataset, 'resample_traj.csv'))
    semantic_df = get_semantic_feature(traj_df)
    temporal_df = get_temporal_feature(traj_df)
    usr_feature_df = pd.merge(semantic_df, temporal_df, on='usr_id', how='inner')
    usr_feature_df = usr_feature_df[['usr_id'] + [col for col in usr_feature_df.columns if col != 'usr_id' and col != 'usr_label']]
    usr_feature_df.to_csv(join(data_dir, dataset, 'usr_feature.csv'), index=False)


def usr_clustering(dataset):
    data_dir = 'cleared_data'
    cluster_num = 8
    seed = 56

    usr_df = pd.read_csv(join(data_dir, dataset, 'usr_feature.csv'))
    use_cols = [col for col in usr_df.columns if col != 'usr_id']
    usr_df = usr_df[['usr_id'] + use_cols]

    model = KMeans(n_clusters=cluster_num, random_state=seed)
    usr_df['label'] = model.fit_predict(usr_df[use_cols])

    usr_df.to_csv(join(data_dir, dataset, 'usr_feature.csv'), index=False)


if __name__ == '__main__':
    extract_usr_pref('fsq_global')
    usr_clustering('fsq_global')
