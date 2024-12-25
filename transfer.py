import ast
import pandas as pd
import transbigdata as tbd


traj_df = pd.read_csv('traj_1.csv')
traj_df['lon'] = traj_df['coordinates'].apply(lambda coord: ast.literal_eval(coord)[0])
traj_df['lat'] = traj_df['coordinates'].apply(lambda coord: ast.literal_eval(coord)[1])
traj_df.to_csv('ori_traj.csv', index=False)

gcj_df = traj_df.apply(lambda row: tbd.gcj02towgs84(row['lon'], row['lat']), axis=1, result_type='expand')
gcj_df.columns = ['lon', 'lat']
gcj_df['id'] = range(len(gcj_df))
gcj_df.to_csv('gcj_traj.csv', index=False)

bd_df = traj_df.apply(lambda row: tbd.bd09towgs84(row['lon'], row['lat']), axis=1, result_type='expand')
bd_df.columns = ['lon', 'lat']
bd_df['id'] = range(len(gcj_df))
bd_df.to_csv('bd_traj.csv', index=False)
