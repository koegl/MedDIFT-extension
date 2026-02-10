from pathlib import Path

import numpy as np

path_ori = "/home/iml/fryderyk.koegl/data/LungCT/keypointsTs/LungCT_{}_0001.csv"
path_pred = "/home/iml/fryderyk.koegl/data/LungCT/predicted_keypoints/LungCT_{}_predicted_0001.csv"

all_distances = []
case_averages = []

for i in range(21, 30):
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

print("Number of images evaluated: ", len(case_averages))

# convert to numpy array for easier handling
all_distances = np.array(all_distances)
case_averages = np.array(case_averages)

mean_error_cases = np.mean(case_averages)
mean_error_keypoints = np.mean(all_distances)

print(f"Mean error across cases: {mean_error_cases:.2f} mm")
print(f"Mean error across all keypoints: {mean_error_keypoints:.2f} mm")

# print standard deviation for additional insight
std_error_cases = np.std(case_averages)
std_error_keypoints = np.std(all_distances)

print(f"Standard deviation across cases: {std_error_cases:.2f} mm")
print(f"Standard deviation across all keypoints: {std_error_keypoints:.2f} mm")

x = 0
