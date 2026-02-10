import subprocess
import sys
from pathlib import Path


SCRIPT = Path(
    "/home/iml/fryderyk.koegl/code/MedDIFT-extension/featurizer.py"
)

IMAGES_DIR = Path("/home/iml/fryderyk.koegl/data/LungCT/imagesTs")
KEYPOINTS_DIR = Path("/home/iml/fryderyk.koegl/data/LungCT/keypointsTs")
OUT_DIR = Path("/home/iml/fryderyk.koegl/data/LungCT/predicted_keypoints")

CONFIG = "config_ddpm_lung"
T = 20
LEVELS = "0123"
BATCH_SIZE = 512
DEVICE = "cuda:0"

START_CASE = 21
END_CASE = 30


def main() -> None:
    """Run diffusion keypoint matching for LungCT_0001..0020."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SCRIPT.exists():
        raise FileNotFoundError(f"Script not found: {SCRIPT}")

    for case_id in range(START_CASE, END_CASE + 1):
        case = f"LungCT_{case_id:04d}"

        img1 = IMAGES_DIR / f"{case}_0000.nii.gz"
        img2 = IMAGES_DIR / f"{case}_0001.nii.gz"
        lm1 = KEYPOINTS_DIR / f"{case}_0000.csv"
        out_csv = OUT_DIR / f"{case}_predicted_0001.csv"

        if not img1.exists():
            print(f"[SKIP] Missing img1: {img1}")
            continue
        if not img2.exists():
            print(f"[SKIP] Missing img2: {img2}")
            continue
        if not lm1.exists():
            print(f"[SKIP] Missing lm1:  {lm1}")
            continue

        cmd = [
            sys.executable,
            str(SCRIPT),
            "--img1",
            str(img1),
            "--img2",
            str(img2),
            "--lm1",
            str(lm1),
            "--out",
            str(out_csv),
            "--config",
            CONFIG,
            "--t",
            str(T),
            "--levels",
            LEVELS,
            "--batch_size",
            str(BATCH_SIZE),
            "--device",
            DEVICE,
        ]

        print(f"\n[RUN] {case}")
        print(" ".join(cmd))

        subprocess.run(cmd, check=True)

    print("\n[DONE] Finished all cases.")


if __name__ == "__main__":
    main()
