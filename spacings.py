from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> None:
    """Print voxel spacing (sx, sy, sz) for every NIfTI file in a folder."""
    root = Path("/home/iml/fryderyk.koegl/data/LungCT/imagesTs")

    paths = sorted(
        [*root.rglob("*.nii"), *root.rglob("*.nii.gz")],
        key=lambda p: p.name,
    )

    if not paths:
        raise RuntimeError(f"No NIfTI files found in: {root}")

    spacings = set()
    for p in paths:
        img = nib.load(str(p))
        spacing = tuple(np.round(img.header.get_zooms()[:3], 6))

        spacing = [float(s) for s in spacing]

        spacings.add(tuple(spacing))
        print(f"{p.name}\t{spacing}")

    x = 0


if __name__ == "__main__":
    main()
