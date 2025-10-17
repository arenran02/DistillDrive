import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
import mmcv

# ====== 하이퍼파라미터 ======
K = 6          # per-command 클러스터 수(최종 저장 형태는 고정 유지: (3, K, STEPS, 2))
STEPS = 6      # 사용 스텝 수 (gt_ego_fut_trajs가 6스텝 기준)
MIN_STEPS = 3  # 마스크 합이 이보다 작으면 제외

# 당신 환경에 맞게 확인 (예: data/nuscenes/mini/...)
fp = 'data/infos/mini/nuscenes_infos_train.pkl'

os.makedirs('vis/kmeans', exist_ok=True)
os.makedirs('data/kmeans', exist_ok=True)

data = mmcv.load(fp)
data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))

# cmd: 0=Right, 1=Left, 2=Straight (one-hot argmax)
navi_trajs = [[], [], []]  # 각 원소는 (STEPS, 2) 절대궤적

for info in tqdm(data_infos):
    if 'gt_ego_fut_trajs' not in info:
        continue
    deltas = info['gt_ego_fut_trajs']          # (T, 2) - 프레임 간 오프셋
    mask = info.get('gt_ego_fut_masks', None)  # (T,)
    if mask is None:
        T = deltas.shape[0]
    else:
        T = int(np.round(mask.sum()))
    if T < MIN_STEPS:
        continue

    # 길이 정규화: [0:T]만 쓰고 부족하면 0으로 패딩, 길면 자르기
    traj = deltas[:T]
    if T < STEPS:
        pad = np.zeros((STEPS - T, 2), dtype=traj.dtype)
        traj = np.concatenate([traj, pad], axis=0)
    else:
        traj = traj[:STEPS]

    # 절대궤적(누적합)으로 변환
    abs_traj = traj.cumsum(axis=0)  # (STEPS, 2)

    cmd = int(info['gt_ego_fut_cmd'].astype(np.int32).argmax(axis=-1))
    navi_trajs[cmd].append(abs_traj)

def cluster_group(trajs2d, K, steps):
    """trajs2d: List[(steps,2)] -> (K, steps, 2)"""
    if len(trajs2d) == 0:
        # 해당 명령에 데이터가 없으면 0으로 채워 고정 형태 유지
        return np.zeros((K, steps, 2), dtype=np.float32)

    X = np.array(trajs2d, dtype=np.float32).reshape(len(trajs2d), steps * 2)

    # 너무 중복되면 KMeans가 수렴 경고 -> 중복 제거로 안정화
    X = np.unique(np.round(X, 4), axis=0)

    K_eff = min(K, len(X))  # 샘플보다 큰 K 방지
    km = KMeans(n_clusters=K_eff, n_init=10, random_state=0)
    centers = km.fit(X).cluster_centers_.reshape(-1, steps, 2)

    # K_eff < K이면 반복 복제로 패딩해 (K, steps, 2) 형태 고정
    if K_eff < K:
        reps = int(np.ceil(K / K_eff))
        centers = np.tile(centers, (reps, 1, 1))[:K]

    return centers.astype(np.float32)

clusters = []
plt.figure()
for g in range(3):
    centers = cluster_group(navi_trajs[g], K, STEPS)
    clusters.append(centers)
    # 시각화
    for j in range(K):
        plt.scatter(centers[j, :, 0], centers[j, :, 1])
plt.savefig(f'vis/kmeans/plan_{K}.png', bbox_inches='tight')
plt.close()

clusters = np.stack(clusters, axis=0)  # (3, K, STEPS, 2)
np.save(f'data/kmeans/kmeans_plan_{K}.npy', clusters)
print('saved:', clusters.shape)