from typing import Tuple

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    EnsureTyped,
    ScaleIntensityRanged,
    Resized
)
import numpy as np
import torch


def round_number(number: int, base: int = 128) -> int:
    return int(max(round(number / base), 1) * base)


def round_number_up(number: int, base: int = 128) -> int:
    return ((number + base - 1) // base) * base


def preprocess_img_l2reg(
        nifti_path: str, win_level: int = 0, win_width: int = 2000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int]]:
    a_min = win_level - win_width / 2
    a_max = win_level + win_width / 2
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
