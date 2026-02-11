import re
from pathlib import Path
import warnings
import os

from typing import Iterable

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from nibabel.orientations import aff2axcodes

def rename_cases_zfill(width: int = 4, dry_run: bool = True) -> None:
    """Rename files like case10_image1.nii -> case0010_image1.nii."""
    root = Path("/home/iml/fryderyk.koegl/data/Lung-DIR-QA/nii-txt")

    pattern = re.compile(r"^(case)(\d+)(_.+)$")

    renames: list[tuple[Path, Path]] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue

        m = pattern.match(p.name)
        if not m:
            continue

        prefix, num_str, suffix = m.groups()
        new_name = f"{prefix}{int(num_str):0{width}d}{suffix}"
        new_path = p.with_name(new_name)

        if new_path == p:
            continue

        if new_path.exists():
            raise FileExistsError(f"Target already exists: {new_path}")

        renames.append((p, new_path))

    for old, new in renames:
        print(f"{old.name}  ->  {new.name}")
        if not dry_run:
            old.rename(new)



from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk


def resample_to_half(input_dir: str | Path, factor: float) -> Path:
    """Downsample NIfTI volumes by 2× per axis and scale voxel-space landmarks accordingly."""
    in_dir = Path(input_dir)
    out_dir = in_dir.parent / f"{in_dir.name}_half"
    out_dir.mkdir(parents=True, exist_ok=True)

    nii_paths: Iterable[Path] = sorted(list(in_dir.glob("*.nii")) + list(in_dir.glob("*.nii.gz")))
    for nii_path in nii_paths:
        img = sitk.ReadImage(str(nii_path))
        old_size = np.array(list(img.GetSize()), dtype=np.int64)
        new_size = tuple(np.maximum(1, (old_size // factor).astype(np.int64)).tolist())
        new_spacing = tuple((np.array(list(img.GetSpacing()), dtype=np.float64) * factor).tolist())

        resampled = sitk.Resample(
            img,
            new_size,
            sitk.Transform(),
            sitk.sitkLinear,
            img.GetOrigin(),
            new_spacing,
            img.GetDirection(),
            0.0,
            img.GetPixelID(),
        )

        sitk.WriteImage(resampled, str(out_dir / nii_path.name))

        
        txt_in = in_dir / f"{nii_path.stem.replace("image", "landmarks")}.txt"
        if txt_in.exists():
            pts = np.loadtxt(txt_in, delimiter=",", dtype=np.float64)
            pts = np.atleast_2d(pts)
            pts[:, :3] /= factor
            np.savetxt(out_dir / txt_in.name, pts, delimiter=",", fmt="%.6f")

    return out_dir





if __name__ == "__main__":
    # rename_cases_zfill(width=4, dry_run=False)
    resample_to_half("/home/iml/fryderyk.koegl/data/Lung-DIR-QA/nii-txt", factor=3.0)

/home/iml/fryderyk.koegl/LungCT_dir_qa/case0001_predicted.csv


