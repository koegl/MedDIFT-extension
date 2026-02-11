import argparse


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for full-dataset execution."""
    p = argparse.ArgumentParser(description="Diffusion feature-based keypoint matching (full datasets).")

    p.add_argument("--config", type=str, default="config_ddpm_lung", help="MAISI config name/path.")
    p.add_argument("--t", type=int, default=20, help="Diffusion timestep.")
    p.add_argument("--levels", type=str, default="0123", help="Feature levels to use, e.g. 0123.")
    p.add_argument("--batch_size", type=int, default=128, help="Landmark batch size.")
    p.add_argument("--device", type=str, default="cuda:0", help="Torch device string.")

    p.add_argument(
        "--lungct_images_dir",
        type=str,
        default="/home/iml/fryderyk.koegl/data/LungCT/imagesTr",
        help="LungCT Learn2Reg images directory.",
    )
    p.add_argument(
        "--lungct_keypoints_dir",
        type=str,
        default="/home/iml/fryderyk.koegl/data/LungCT/keypointsTr",
        help="LungCT Learn2Reg keypoints directory.",
    )


    p.add_argument(
        "--dirqa_root_dir",
        type=str,
        default="/home/iml/fryderyk.koegl/data/Lung-DIR-QA/nii-txt_half",
        help="Root directory for LungCT-DIR-QA nii/txt files.",
    )

    p.add_argument(
        "--out_dir",
        type=str,
        default="/home/iml/fryderyk.koegl/data/dift_predictions",
        help="Output directory for LungCT-DIR-QA predictions.",
    )

    return p.parse_args()
