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
    traj_df = pd.read_csv(join(data_dir, dataset, 'traj.csv'))
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
    # use_cols = [
    #     'bank', 'bar', 'building', 'education', 'entertainment', 'food', 'hotel', 'medical', 'office',
    #     'other', 'outdoor', 'park', 'residential', 'service', 'sports', 'store', 'theater', 'transportation'
    # ]
    use_cols = [col for col in usr_df.columns if col != 'usr_id']
    usr_df = usr_df[['usr_id'] + use_cols]

    model = KMeans(n_clusters=cluster_num, random_state=seed)
    usr_df['label'] = model.fit_predict(usr_df[use_cols])

    usr_df.to_csv(join(data_dir, dataset, 'usr_feature.csv'), index=False)


def visualize_usr(dataset):
    data_dir = 'cleared_data'
    save_dir = join(data_dir, dataset, 'visual')
    os.makedirs(save_dir, exist_ok=True)

    usr_df = pd.read_csv(join(data_dir, dataset, 'usr_feature.csv'))
    # feat_cols = [col for col in usr_df.columns if col != 'usr_id' and col != 'label' and not any(char.isdigit() for char in col)]
    feat_cols = [col for col in usr_df.columns if any(char.isdigit() for char in col)]

    plt.rcParams.update({'font.size': 10})
    # 创建 2 行 4 列的子图布局，并设置图像大小
    fig, axes = plt.subplots(2, 4, figsize=(16, 12))
    for idx, (usr_label, group) in enumerate(usr_df.groupby('label')):
        ax = axes.flat[idx]
        feat_val = {col: group[col].mean() for col in feat_cols}
        keys = list(feat_val.keys())
        vals = list(feat_val.values())
        ax.bar(keys, vals, color='skyblue')  # 绘制柱状图
        ax.set_title(f"User {idx}", fontsize=16)
        # ax.set_xlabel("Categories", fontsize=12)
        ax.set_ylabel("TF-IDF", fontsize=12)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=45, ha='right')

        # 设置纵横比，使每个子图的纵轴比例合适
        ax.set_aspect(aspect='auto')  # 可以尝试 'equal' 或具体比例值

    # 调整布局，减少列间距，增加行间距
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.2, hspace=0.4, wspace=0.2)
    plt.savefig(join(save_dir, 'usr_feature_time.pdf'), format='pdf')


if __name__ == '__main__':
    extract_usr_pref('fsq_global')
    usr_clustering('fsq_global')
    # visualize_usr('fsq_global')
