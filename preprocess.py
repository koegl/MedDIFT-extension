from typing import Tuple

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityRanged,
)
import nibabel as nib
from nibabel.orientations import aff2axcodes
import numpy as np
import torch


def round_number(number: int, base: int = 128) -> int:
    return int(max(round(number / base), 1) * base)


def round_number_up(number: int, base: int = 128) -> int:
    return ((number + base - 1) // base) * base


def preprocess_img_lungct_l2reg(
        nifti_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int]]:
    a_min = -1000
    a_max = 1000
    # print(f"Preprocessing with win level {win_level}HU and win width {win_width}HU...")

    # Load and preprocess image
    img_data = {"image": nifti_path}
    plain_transform = Compose([
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes="RAS"),
        EnsureTyped(keys="image", dtype=torch.float32),
        ScaleIntensityRanged(keys="image", a_min=a_min,
                             a_max=a_max, b_min=0, b_max=1, clip=True),
    ])

    preprocessed = plain_transform(img_data)
    shape = preprocessed["image"].shape[1:]
    print("original image shape WHD xyz: ", shape)

    new_dim = tuple(round_number(x) for x in shape)
    resize = Resized(keys="image", spatial_size=new_dim, mode="trilinear")
    transformed = resize(preprocessed)

    affine = transformed["image"].meta["affine"].numpy()
    original_np = preprocessed["image"].numpy().squeeze()
    image_np = transformed["image"].numpy().squeeze()
    print("pre-processed image shape: ", image_np.shape)

    return original_np, image_np, affine, tuple(shape)


def preprocess_img_lungct_dir_qa(
        nifti_path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int]]:
    a_min = -1000
    a_max = 1000
    # print(f"Preprocessing with win level {win_level}HU and win width {win_width}HU...")

    # Load and preprocess image
    img_data = {"image": nifti_path}
    plain_transform = Compose([
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes="RAS"),
        EnsureTyped(keys="image", dtype=torch.float32),
        Lambdad(keys="image", func=lambda x: x - 1000.0),
        ScaleIntensityRanged(keys="image", a_min=a_min, a_max=a_max, b_min=0, b_max=1, clip=True),
    ])

    preprocessed = plain_transform(img_data)
    shape = preprocessed["image"].shape[1:]
    print("original image shape WHD xyz: ", shape)

    new_dim = tuple(round_number(x) for x in shape)
    resize = Resized(keys="image", spatial_size=new_dim, mode="trilinear")
    transformed = resize(preprocessed)

    affine = transformed["image"].meta["affine"].numpy()
    original_np = preprocessed["image"].numpy().squeeze()
    image_np = transformed["image"].numpy().squeeze()
    print("pre-processed image shape: ", image_np.shape)

    return original_np, image_np, affine, tuple(shape)



def load_landmarks(landmarks_csv_path: str, nifti_path: str) -> np.ndarray:
    """Load Nx3 voxel-index landmarks and convert them to RAS index convention by flipping axes as needed."""
    img = nib.load(nifti_path)
    shape = np.array(img.shape[:3], dtype=np.int64)
    axcodes = aff2axcodes(img.affine)

    pts = np.loadtxt(landmarks_csv_path, delimiter=",", dtype=np.float32)
    pts = np.atleast_2d(pts)
    if pts.shape[1] != 3:
        raise ValueError(f"Expected Nx3 landmarks, got shape {pts.shape}")

    if axcodes[0] == "L":
        pts[:, 0] = (shape[0] - 1) - pts[:, 0]
    elif axcodes[0] != "R":
        raise ValueError(f"Unsupported x axis code {axcodes[0]} in {axcodes}")

    if axcodes[1] == "P":
        pts[:, 1] = (shape[1] - 1) - pts[:, 1]
    elif axcodes[1] != "A":
        raise ValueError(f"Unsupported y axis code {axcodes[1]} in {axcodes}")

    if axcodes[2] == "I":
        pts[:, 2] = (shape[2] - 1) - pts[:, 2]
    elif axcodes[2] != "S":
        raise ValueError(f"Unsupported z axis code {axcodes[2]} in {axcodes}")

    return pts

if __name__ == "__main__":
    path_image_test = "/home/iml/fryderyk.koegl/data/Lung-DIR-QA/nii-txt_half/case0001_image1.nii"
    path_image_test_trans = "/home/iml/fryderyk.koegl/data/Lung-DIR-QA/nii-txt_half/case0001_image1_trans.nii"

    path_lm_test = "/home/iml/fryderyk.koegl/data/Lung-DIR-QA/nii-txt_half/case0001_landmarks1.txt"
    path_lm_test_trans = "/home/iml/fryderyk.koegl/data/Lung-DIR-QA/nii-txt_half/case0001_landmarks1_trans.txt"

    original_np, image_np, affine, shape = preprocess_img_lungct_dir_qa(path_image_test)
    image_trans = nib.nifti1.Nifti1Image(image_np, affine)
    nib.save(image_trans, path_image_test_trans)

    landmarks = load_landmarks(path_lm_test, path_image_test)
    np.savetxt(path_lm_test_trans, landmarks, delimiter=",", fmt="%.2f")

    x = 0
    
      