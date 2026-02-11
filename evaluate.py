from pathlib import Path

from typing import Tuple

import numpy as np

def evaluate_lungct_dir_qa() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    path_ori_first = "/home/iml/fryderyk.koegl/data/Lung-DIR-QA/nii-txt_half/case{}_landmarks1.txt"
    path_ori_second = "/home/iml/fryderyk.koegl/data/Lung-DIR-QA/nii-txt_half/case{}_landmarks2.txt"
    path_prediction = "/home/iml/fryderyk.koegl/datadift_predictions/LungCT_dir_qa/case{}_predicted.csv"

    case_averages_after = []
    all_distances_before = []
    case_averages_before = []
    all_distances_after = []
    case_averages_after = []

    for i in range(1, 2):
        path_ori_first_i = Path(path_ori_first.format(str(i).zfill(4)))
        path_ori_second_i = Path(path_ori_second.format(str(i).zfill(4)))
        path_pred_i = Path(path_prediction.format(str(i).zfill(4)))

        if not (path_ori_first_i.exists() and path_ori_second_i.exists() and path_pred_i.exists()):
            print(f"Missing files for case {i:04d}, skipping.")
            continue

        ori_first = np.loadtxt(path_ori_first_i, delimiter=",")
        ori_second = np.loadtxt(path_ori_second_i, delimiter=",")
        pred = np.loadtxt(path_pred_i, delimiter=",")

        # calculate the l2 distance between the two sets of points
        distances_before = np.linalg.norm(ori_first - ori_second, axis=1)
        dist_avg_before = np.mean(distances_before)

        distances_after = np.linalg.norm(ori_second - pred, axis=1)
        dist_avg_after = np.mean(distances_after)

        all_distances_before.extend(distances_before)
        case_averages_before.append(dist_avg_before)
        all_distances_after.extend(distances_after)
        case_averages_after.append(dist_avg_after)

    return all_distances_before, case_averages_before, all_distances_after, case_averages_after
    
def evaluate_lungct_l2reg() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path_ori_first = "/home/iml/fryderyk.koegl/data/LungCT/keypointsTs/LungCT_{}_0000.csv"
    path_ori_second = "/home/iml/fryderyk.koegl/data/LungCT/keypointsTs/LungCT_{}_0001.csv"
    path_prediction = "/home/iml/fryderyk.koegl/data/LungCT/predicted_keypoints_ori/LungCT_{}_predicted_0001.csv"

    all_distances_before = []
    case_averages_before = []
    all_distances_after = []
    case_averages_after = []

    spacing = np.array([1.75, 1.25, 1.75], dtype=np.float64)

    # before
    for i in range(21, 30):
        path_first_i = Path(path_ori_first.format(str(i).zfill(4)))
        path_second_i = Path(path_ori_second.format(str(i).zfill(4)))

        if not (path_first_i.exists() and path_second_i.exists()):
            print(f"Missing files for case {i:04d}, skipping.")
            continue

        ori = np.loadtxt(path_first_i, delimiter=",")
        pred = np.loadtxt(path_second_i, delimiter=",")

        # calculate the l2 distance between the two sets of points
        diff_mm = (ori - pred) * spacing[None, :]
        distances = np.linalg.norm(diff_mm, axis=1)
        dist_avg = np.mean(distances)

        all_distances_before.extend(distances)
        case_averages_before.append(dist_avg)

    # after
    for i in range(21, 30):
        path_ori_i = Path(path_ori_second.format(str(i).zfill(4)))
        path_pred_i = Path(path_prediction.format(str(i).zfill(4)))

        if not (path_ori_i.exists() and path_pred_i.exists()):
            print(f"Missing files for case {i:04d}, skipping.")
            continue

        ori = np.loadtxt(path_ori_i, delimiter=",")
        pred = np.loadtxt(path_pred_i, delimiter=",")

        # calculate the l2 distance between the two sets of points
        diff_mm = (ori - pred) * spacing[None, :]
        distances = np.linalg.norm(diff_mm, axis=1)
        dist_avg = np.mean(distances)

        all_distances_after.extend(distances)
        case_averages_after.append(dist_avg)

    return all_distances_before, case_averages_before, all_distances_after, case_averages_after

all_distances_before, case_averages_before, all_distances_after, case_averages_after = evaluate_lungct_l2reg()


print("Number of images evaluated: ", len(case_averages_after))

# convert to numpy array for easier handling
all_distances_before = np.array(all_distances_before)
case_averages_before = np.array(case_averages_before)
all_distances_after = np.array(all_distances_after)
case_averages_after = np.array(case_averages_after)


mean_error_cases_bef = np.nanmean(case_averages_before)
mean_error_keypoints_bef = np.nanmean(all_distances_before)
mean_error_cases_af = np.nanmean(case_averages_after)
mean_error_keypoints_af = np.nanmean(all_distances_after)

print(f"Mean error across cases (before): {mean_error_cases_bef:.2f} mm")
print(f"Mean error across cases (after): {mean_error_cases_af:.2f} mm")
print(f"Mean error across all keypoints (before): {mean_error_keypoints_bef:.2f} mm")
print(f"Mean error across all keypoints (after): {mean_error_keypoints_af:.2f} mm")
print()
# print standard deviation for additional insight
std_error_cases_bef = np.nanstd(case_averages_before)
std_error_keypoints_bef = np.nanstd(all_distances_before)
std_error_cases_af = np.nanstd(case_averages_after)
std_error_keypoints_af = np.nanstd(all_distances_after)


print(f"Standard deviation across cases (before): {std_error_cases_bef:.2f} mm")
print(f"Standard deviation across cases (after): {std_error_cases_af:.2f} mm")
print(f"Standard deviation across all keypoints (before): {std_error_keypoints_bef:.2f} mm")
print(f"Standard deviation across all keypoints (after): {std_error_keypoints_af:.2f} mm")

x = 0
