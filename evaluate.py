from pathlib import Path

import numpy as np

path_ori = "/home/iml/fryderyk.koegl/data/LungCT/keypointsTr/LungCT_{}_0001.csv"
path_pred = "/home/iml/fryderyk.koegl/data/LungCT/predicted_keypoints/LungCT_{}_predicted_0001.csv"

all_distances = []
case_averages = []

for i in range(1, 21):
    path_ori_i = Path(path_ori.format(str(i).zfill(4)))
    path_pred_i = Path(path_pred.format(str(i).zfill(4)))

    if not (path_ori_i.exists() and path_pred_i.exists()):
        print(f"Missing files for case {i:04d}, skipping.")
        continue

    ori = np.loadtxt(path_ori_i, delimiter=",")
    pred = np.loadtxt(path_pred_i, delimiter=",")

    # calculate the l2 distance between the two sets of points
    distances = np.linalg.norm(ori - pred, axis=1)
    dist_avg = np.mean(distances)

    all_distances.extend(distances)
    case_averages.append(dist_avg)

# convert to numpy array for easier handling
all_distances = np.array(all_distances)
case_averages = np.array(case_averages)

mean_error_cases = np.mean(case_averages)
mean_error_keypoints = np.mean(all_distances)

x = 0
