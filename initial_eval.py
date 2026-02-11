import re
from pathlib import Path
from typing import Dict, Tuple

import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine


def load_landmarks_csv(path: Path) -> np.ndarray:
    """Load Nx3 landmarks from a headerless CSV."""
    pts = np.loadtxt(path, delimiter=",", dtype=np.float64)
    pts = np.atleast_2d(pts)
    if pts.shape[1] != 3:
        raise ValueError(f"{path}: expected Nx3, got {pts.shape}")
    return pts


def find_pairs(root: Path) -> Dict[str, Tuple[Path, Path]]:
    """Find (case_id -> (0000.csv, 0001.csv)) pairs in a directory."""
    pattern = re.compile(r"^(LungCT_\d{4})_(0000|0001)\.csv$")
    groups: Dict[str, Dict[str, Path]] = {}
    for p in root.glob("*.csv"):
        m = pattern.match(p.name)
        if not m:
            continue
        case_id, which = m.group(1), m.group(2)
        groups.setdefault(case_id, {})[which] = p

    pairs: Dict[str, Tuple[Path, Path]] = {}
    for case_id, d in groups.items():
        if "0000" in d and "0001" in d:
            pairs[case_id] = (d["0000"], d["0001"])
    return dict(sorted(pairs.items(), key=lambda x: x[0]))


def load_nifti(case_id: str, suffix: str, root: Path) -> nib.Nifti1Image:
    """Load the matching NIfTI for a case_id and suffix (0000/0001)."""
    candidates = [
        root / f"{case_id}_{suffix}.nii.gz",
        root / f"{case_id}_{suffix}.nii",
    ]
    for p in candidates:
        if p.exists():
            return nib.load(str(p))
    raise FileNotFoundError(f"Missing NIfTI for {case_id}_{suffix} in {root}")


def main() -> None:
    """Compute TRE in mm using NIfTI affines (world-space)."""
    csv_root = Path("/home/iml/fryderyk.koegl/data/LungCT/keypointsTs")
    img_root = Path("/home/iml/fryderyk.koegl/data/LungCT/imagesTs")

    pairs = find_pairs(csv_root)
    if not pairs:
        raise RuntimeError(f"No CSV pairs found in {csv_root}")

    case_means: list[float] = []
    all_tre: list[np.ndarray] = []

    for case_id, (p0_csv, p1_csv) in pairs.items():
        pts0 = load_landmarks_csv(p0_csv)
        pts1 = load_landmarks_csv(p1_csv)

        img0 = load_nifti(case_id, "0000", img_root)
        img1 = load_nifti(case_id, "0001", img_root)

        w0 = apply_affine(img0.affine, pts0)
        w1 = apply_affine(img1.affine, pts1)

        tre = np.linalg.norm(w0 - w1, axis=1)
        case_means.append(float(tre.mean()))
        all_tre.append(tre)

    all_tre_arr = np.concatenate(all_tre, axis=0)

    print(f"Cases\t{len(case_means)}")
    print(f"CASE_AVG\t{all_tre_arr.size}\t{np.mean(case_means)}")
    print(f"ALL_POINTS\t{all_tre_arr.size}\t{float(all_tre_arr.mean())}")
    x = 0


if __name__ == "__main__":
    main()
