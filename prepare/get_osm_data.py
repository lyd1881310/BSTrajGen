import ast
import json
import os
import math
import osmnx as ox
from os.path import join

import yaml
from geopy.distance import great_circle
import logging

import pandas as pd

# https://overpass-api.de/api
# ox.config(log_console=True, use_cache=True, log_level=logging.DEBUG,
#           overpass_endpoint='https://overpass.kumi.systems/api/interpreter')
# ox.config(log_console=True, use_cache=True, log_level=logging.DEBUG)

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s](%(asctime)s):%(message)s',
    datefmt='%H:%M:%S'
)
file_handler = logging.FileHandler('download_log.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(levelname)s](%(asctime)s):%(message)s')
file_handler.setFormatter(formatter)
logging.getLogger().addHandler(file_handler)


def bbox_partition(min_lon, min_lat, max_lon, max_lat, max_len=10*1000):
    """
    切分成更小的栅格再发送请求
    """
    distance = great_circle((min_lat, min_lon), (max_lat, max_lon)).meters
    y_num = x_num = math.ceil(distance / max_len)
    d_lon = (max_lon - min_lon) / x_num
    d_lat = (max_lat - min_lat) / y_num
    cell_coords = []
    for i in range(x_num):
        for j in range(y_num):
            cell_coords.append({
                'min_lon': min_lon + i * d_lon,
                'min_lat': min_lat + j * d_lat,
                'max_lon': min_lon + (i + 1) * d_lon,
                'max_lat': min_lat + (j + 1) * d_lat
            })
    return cell_coords


def download_road(city):
    print(f'Download road {city} ...... ')
    data_feat = json.load(open(join('cleared_data', city, 'data_feature.json'), 'r'))
    bbox = (data_feat['max_lat'], data_feat['min_lat'], data_feat['max_lon'], data_feat['min_lon'])
    graph = ox.graph_from_bbox(bbox=bbox, network_type='drive')
    ox.save_graphml(graph, join('download', city, 'osm_road.osm'))


def download_osm_entity(city, tag):
    print(f'Download osm entity {city} ...... ')
    os.makedirs(join('download', city), exist_ok=True)

    data_feat = yaml.safe_load(open(join('cleared_data', city, 'bound.yaml'), 'r'))
    sub_bboxes = bbox_partition(
        min_lon=data_feat['min_lon'], min_lat=data_feat['min_lat'],
        max_lon=data_feat['max_lon'], max_lat=data_feat['max_lat'],
        max_len=15*1000
    )
    logging.info(f'{city} sub bboxes {len(sub_bboxes)}')
    poi_dfs = []
    for itr, sub_bbox in enumerate(sub_bboxes):
        bbox_param = (sub_bbox['max_lat'], sub_bbox['min_lat'], sub_bbox['max_lon'], sub_bbox['min_lon'])
        try:
            feature_gdf = ox.features_from_bbox(bbox=bbox_param, tags={tag: True})
        except Exception as e:
            logging.info(f'Exception {itr} {city} ...... \n {e}', )
            continue
        if len(feature_gdf) == 0:
            continue
        poi_df = feature_gdf[['name', tag, 'geometry']].reset_index()
        poi_df['tag'] = tag
        poi_df = poi_df.rename(columns={tag: 'type'})
        poi_dfs.append(poi_df)
        # 及时保存
        poi_df = pd.concat(poi_dfs)
        logging.info(f'total num {len(poi_df)}')
        poi_df.to_csv(join('download', city, f'osm_{tag}.csv'), index=False)


if __name__ == '__main__':
    download_osm_entity('nbo', 'building')
