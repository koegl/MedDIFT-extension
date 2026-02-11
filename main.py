import faulthandler
import os

from typing import List

from monai.utils import set_determinism
import numpy as np
import SimpleITK as sitk
import torch

import cli
import datasets
import features
import helper
import matching
import preprocess   

from maisi.scripts.diff_model_setting import setup_logging

faulthandler.enable(all_threads=True)
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logger = setup_logging("diffusion_feature_matching")


def run_job(
    job: datasets.Job,
    cfg,
    autoencoder: torch.nn.Module,
    diffusion_unet: torch.nn.Module,
    scale_factor: torch.Tensor,
    device: torch.device,
    timestep: int,
    levels: str,
    batch_size: int,
    dataset: str,
) -> None:
    """Run one job end-to-end and write predicted landmark CSV."""
    spacing1 = sitk.ReadImage(str(job.img1)).GetSpacing()
    spacing2 = sitk.ReadImage(str(job.img2)).GetSpacing()

    landmarks1 = preprocess.load_landmarks(str(job.lm1), str(job.img1))

    with torch.no_grad():
        feats1, _ = features.extract_features_all_levels(
            cfg, spacing1, job.img1, autoencoder, diffusion_unet, device, timestep, scale_factor, dataset
        )
        feats2, _ = features.extract_features_all_levels(
            cfg, spacing2, job.img2, autoencoder, diffusion_unet, device, timestep, scale_factor, dataset
        )

    preds: List[np.ndarray] = []
    with torch.inference_mode():
        for i in range(0, landmarks1.shape[0], batch_size):
            lm_batch = landmarks1[i : i + batch_size]
            matches = matching.match_points_l2reg(lm_batch, feats1, feats2, levels, device)
            preds.append(matches)
            del matches
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    predicted = np.vstack(preds)
    job.out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(job.out_csv, predicted, delimiter=",", fmt="%.2f")
    print(f"[DONE] {job.out_csv}")


def main() -> None:
    """Entry point."""
    args = cli.parse_args()

    set_determinism(42)
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device(args.device)
    cfg = helper.load_config(args.config)

    autoencoder, diffusion_unet, scale_factor = features.load_models(cfg, device)

    dataloaders = datasets.build_dataloaders(args)

    for dataloader in dataloaders:

        for job in dataloader:
            
            if 'l2reg' in str(dataloader.dataset):
                continue

            print(f"[RUN:{str(dataloader.dataset)}] img1={job.img1.name} img2={job.img2.name} lm1={job.lm1.name}")
            run_job(
                job,
                cfg,
                autoencoder,
                diffusion_unet,
                scale_factor,
                device,
                timestep=int(args.t),
                levels=str(args.levels),
                batch_size=int(args.batch_size),
                dataset=str(dataloader.dataset),
            )


if __name__ == "__main__":
    main()
