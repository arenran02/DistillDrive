import os
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import mmcv

K = 6
class NuscenesDataset():
    def __init__(self):
        super().__init__()
        self.info = mmcv.load('data/infos/nuscenes_infos_train.pkl')
        self.data_infos = list(sorted(self.info["infos"], key=lambda e: e["timestamp"]))
    
    def analyse_data_distribution(self, infos):
        navi_trajs = {cmd: [] for cmd in range(3)}
        data_infos_cmd = {cmd: [] for cmd in range(3)} # Left, Straight, Right
        for idx in tqdm(range(len(infos))):
            info = infos[idx]
            plan_traj = info['gt_ego_fut_trajs'].cumsum(axis=-2)
            cmd = info['gt_ego_fut_cmd'].astype(np.int32)
            cmd = cmd.argmax(axis=-1)
            navi_trajs[cmd].append(plan_traj)
            data_infos_cmd[cmd].append(info)
        
        return navi_trajs, data_infos_cmd

    def balanced_infos_resampling(self):
        
        resampling_info = []
        _, data_infos_cmd = self.analyse_data_distribution(self.data_infos)
        # Stargith not resampling
        resampling_info += data_infos_cmd[2]
        # Caculate Ratio
        total_samples = sum([len(v) for _, v in data_infos_cmd.items()])
        cmd_dist = {k: len(v) / total_samples for k, v in data_infos_cmd.items()}
        ratios = [cmd_dist[2] / v for v in cmd_dist.values()] # use straight as base
        
        for cur_cmd_infos, ratio in zip(list(data_infos_cmd.values()), ratios[:2]):
            resampling_info_tmp = np.random.choice(
                cur_cmd_infos, int(len(cur_cmd_infos) * ratio)
            ).tolist()
            resampling_info_tmp = list(sorted(resampling_info_tmp, key=lambda e: e["timestamp"]))
            memory_timestamp = []
            for idx in range(len(resampling_info_tmp)):
                if len(resampling_info_tmp[idx]["sweeps"]) == 0:
                    if resampling_info_tmp[idx]['timestamp'] not in memory_timestamp:
                        memory_timestamp.append(resampling_info_tmp[idx]['timestamp'])
                    else:
                        resampling_info_tmp[idx]["sweeps"].append([])
            
            resampling_info += resampling_info_tmp


        resampling_info = list(sorted(resampling_info, key=lambda e: e["timestamp"]))
        self.data_infos = resampling_info

        # navi_trajs, data_infos_cmd = self.analyse_data_distribution(resampling_info)
        

        clusters = []
        for trajs in navi_trajs:
            trajs = np.concatenate(trajs, axis=0).reshape(-1, 12)
            cluster = KMeans(n_clusters=K).fit(trajs).cluster_centers_
            cluster = cluster.reshape(-1, 6, 2)
            clusters.append(cluster)
            for j in range(K):
                plt.scatter(cluster[j, :, 0], cluster[j, :,1])
        plt.savefig(f'vis/kmeans/plan_{K}', bbox_inches='tight')
        plt.close()

        clusters = np.stack(clusters, axis=0)
        np.save(f'data/kmeans/kmeans_plan_{K}.npy', clusters)

dataset = NuscenesDataset()
dataset.balanced_infos_resampling()