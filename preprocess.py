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
    """Load Nx3 landmarks (sag,cor,ax index space) and map LAS->RAS by flipping x if needed."""
    img = nib.load(nifti_path)
    x_size = int(img.shape[0])
    axcodes = aff2axcodes(img.affine)

    if landmarks_csv_path.endswith(".txt"):
        pts = np.loadtxt(landmarks_csv_path, dtype=np.float32)
    else:
        pts = np.loadtxt(landmarks_csv_path, delimiter=",", dtype=np.float32)
    pts = np.atleast_2d(pts)
    if pts.shape[1] != 3:
        raise ValueError(f"Expected Nx3 landmarks, got shape {pts.shape}")

    if axcodes == ("L", "A", "S"):
        pts[:, 0] = (x_size - 1) - pts[:, 0]
    elif axcodes == ("R", "A", "S"):
        pass
    else:
        raise ValueError(f"Unsupported orientation {axcodes}; expected LAS or RAS.")

    return pts