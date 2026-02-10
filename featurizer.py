"""
Diffusion Feature-Based Keypoint Matching for 3D Image Pairs

Single entry-point script that can:
1) Run one explicit (img1, img2, lm1) job, OR
2) Run a case range in a directory layout (LungCT_XXXX_0000/0001 pattern)

It loads MAISI models once and iterates over all requested jobs.
"""

import argparse
import faulthandler
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import einops
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from monai.utils import set_determinism

from helper import prepare_tensors, load_config
import preprocess

from maisi.scripts.diff_model_setting import setup_logging
from maisi.scripts.utils import define_instance

faulthandler.enable(all_threads=True)
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logger = setup_logging("diffusion_feature_matching")

image_preprocessing_functions = {
    "LungCT_l2reg": preprocess.preprocess_img_lungct_l2reg,
    "LungCT_dir_qa": preprocess.preprocess_img_lungct_dir_qa,
}


@dataclass(frozen=True)
class Job:
    """One matching job: source image, target image, source landmarks, output path."""
    img1: Path
    img2: Path
    lm1: Path
    out_csv: Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for single-job or batch-case execution."""
    p = argparse.ArgumentParser(description="Diffusion feature-based keypoint matching (single or batch).")

    # mode = p.add_mutually_exclusive_group(required=True)
    # mode.add_argument("--single", action="store_true", help="Run one explicit image pair.")
    # mode.add_argument("--cases", action="store_true", help="Run a LungCT-style case range from directories.")

    p.add_argument("--config", type=str, default="config_ddpm_lung", help="MAISI config name/path.")
    p.add_argument("--t", type=int, default=20, help="Diffusion timestep.")
    p.add_argument("--levels", type=str, default="0123", help="Feature levels to use, e.g. 0123.")
    p.add_argument("--batch_size", type=int, default=128, help="Landmark batch size.")
    p.add_argument("--device", type=str, default="cuda:0", help="Torch device string.")
    p.add_argument("--dataset", type=str, default="LungCT_l2reg", help="Preprocessing key.")

    # --single
    p.add_argument("--img1", type=str, default="/home/iml/fryderyk.koegl/data/LungCT/imagesTr/LungCT_0001_0000.nii.gz",
                   help="Path to source image (single mode).")
    p.add_argument("--img2", type=str, default="/home/iml/fryderyk.koegl/data/LungCT/imagesTr/LungCT_0001_0001.nii.gz",
                   help="Path to target image (single mode).")
    p.add_argument("--lm1", type=str, default="/home/iml/fryderyk.koegl/data/LungCT/keypointsTr/LungCT_0001_0000.csv",
                   help="Path to source landmarks CSV (single mode).")
    p.add_argument("--out", type=str, default="/home/iml/fryderyk.koegl/data/LungCT/keypoints_pred_test/LungCT_0001_predicted_0001.csv",
                   help="Output CSV path (single mode).")

    # --cases
    p.add_argument("--images_dir", type=str, default="/home/iml/fryderyk.koegl/data/LungCT/imagesTr",
                   help="Directory containing imagesTs (cases mode).")
    p.add_argument("--keypoints_dir", type=str, default="/home/iml/fryderyk.koegl/data/LungCT/keypointsTr",
                   help="Directory containing keypointsTs (cases mode).")
    p.add_argument("--out_dir", type=str, default="/home/iml/fryderyk.koegl/data/LungCT/keypoints_pred_test",
                    help="Output directory (cases mode).")
    p.add_argument("--start_case", type=int, default=1, help="Start case id (inclusive).")
    p.add_argument("--end_case", type=int, default=1, help="End case id (inclusive).")
    p.add_argument("--case_prefix", type=str, default="LungCT_", help="Case prefix, default 'LungCT_'.")
    p.add_argument("--img1_suffix", type=str, default="_0000.nii.gz", help="Suffix for source image file.")
    p.add_argument("--img2_suffix", type=str, default="_0001.nii.gz", help="Suffix for target image file.")
    p.add_argument("--lm1_suffix", type=str, default="_0000.csv", help="Suffix for source landmark file.")
    p.add_argument("--out_suffix", type=str, default="_predicted_0001.csv", help="Suffix for output CSV file.")

    return p.parse_args()


def validate_dataset_key(dataset: str) -> None:
    """Validate that the chosen preprocessing dataset key exists."""
    if dataset not in image_preprocessing_functions:
        raise ValueError(
            f"Unsupported dataset: {dataset}. Supported: {sorted(image_preprocessing_functions.keys())}"
        )


def make_single_job(args: argparse.Namespace) -> Job:
    """Build a single explicit job from args."""
    missing = [k for k in ("img1", "img2", "lm1", "out") if getattr(args, k) is None]
    if missing:
        raise ValueError(f"--single requires: {', '.join('--' + m for m in missing)}")
    return Job(Path(args.img1), Path(args.img2), Path(args.lm1), Path(args.out))


def iter_case_jobs(args: argparse.Namespace) -> Iterable[Job]:
    """Yield jobs for a LungCT-style case range."""
    for k in ("images_dir", "keypoints_dir", "out_dir"):
        if getattr(args, k) is None:
            raise ValueError(f"--cases requires --{k}")

    images_dir = Path(args.images_dir)
    keypoints_dir = Path(args.keypoints_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for case_id in range(args.start_case, args.end_case + 1):
        case = f"{args.case_prefix}{case_id:04d}"
        img1 = images_dir / f"{case}{args.img1_suffix}"
        img2 = images_dir / f"{case}{args.img2_suffix}"
        lm1 = keypoints_dir / f"{case}{args.lm1_suffix}"
        out_csv = out_dir / f"{case}{args.out_suffix}"

        if not img1.exists():
            print(f"[SKIP] Missing img1: {img1}")
            continue
        if not img2.exists():
            print(f"[SKIP] Missing img2: {img2}")
            continue
        if not lm1.exists():
            print(f"[SKIP] Missing lm1:  {lm1}")
            continue

        yield Job(img1=img1, img2=img2, lm1=lm1, out_csv=out_csv)


def load_models(
    cfg: argparse.Namespace, device: torch.device
) -> Tuple[torch.nn.Module, torch.nn.Module, torch.Tensor]:
    """Load autoencoder, diffusion UNet, and scale_factor once."""
    autoencoder = define_instance(cfg, "autoencoder_def").to(device)
    autoencoder.load_state_dict(torch.load(cfg.trained_autoencoder_path, map_location=device))
    autoencoder.eval()

    diffusion_unet = define_instance(cfg, "diffusion_unet_def").to(device)
    ckpt = torch.load(cfg.trained_diffusion_path, weights_only=False)
    diffusion_unet.load_state_dict(ckpt["unet_state_dict"], strict=False)
    diffusion_unet.eval()

    return autoencoder, diffusion_unet, ckpt["scale_factor"].to(device)


def extract_latent(
    image_np: np.ndarray,
    autoencoder: torch.nn.Module,
    scale_factor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Encode a preprocessed 3D image into latent space (W,H,D,C) numpy."""
    autoencoder.eval()
    with torch.inference_mode(), torch.amp.autocast("cuda"):
        x = torch.from_numpy(image_np).float().to(device)[None, None]
        latent = (autoencoder.encode_stage_2_inputs(x) * scale_factor).float()
    return latent.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()


@torch.no_grad()
def add_and_denoise_one_step_ddpm(
    latent_np: np.ndarray,
    diffusion_model: torch.nn.Module,
    modality_tensor: torch.Tensor,
    spacing_tensor: torch.Tensor,
    noise_scheduler,
    device: torch.device,
    timestep: int,
    top_region_index_tensor: torch.Tensor,
    bottom_region_index_tensor: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Run one noisy step + one UNet forward pass and return extracted feature dict."""
    latent = torch.from_numpy(latent_np).permute(3, 0, 1, 2)[None].to(device)
    noise = torch.randn_like(latent)
    noisy = noise_scheduler.add_noise(latent, noise, torch.tensor([timestep], device=device))

    with torch.amp.autocast("cuda"):
        _ = diffusion_model(
            x=noisy,
            timesteps=torch.tensor([timestep], device=device),
            class_labels=modality_tensor,
            spacing_tensor=spacing_tensor,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            extract_feature=True,
        )
    return diffusion_model.features


def upsample_to_original(features: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
    """Upsample 3D feature maps (C,W,H,D) to (C,W,H,D) at target spatial size."""
    x = features.permute(0, 3, 2, 1)[None].cpu().float()
    w, h, d = target_shape
    y = F.interpolate(x, size=(d, h, w), mode="trilinear", align_corners=False)
    return y.squeeze(0).permute(0, 3, 2, 1)


def extract_features_all_levels(
    cfg: argparse.Namespace,
    spacing: Tuple[float, float, float],
    image_path: Path,
    autoencoder: torch.nn.Module,
    diffusion_unet: torch.nn.Module,
    device: torch.device,
    timestep: int,
    scale_factor: torch.Tensor,
    dataset: str,
) -> Tuple[List[torch.Tensor], Tuple[int, int, int]]:
    """Preprocess image, run DDPM feature extraction, upsample all UNet decoder levels."""
    _, pre_np, _, original_shape = image_preprocessing_functions[dataset](str(image_path))
    top_t, bot_t, spacing_t, mod_t = prepare_tensors(cfg, spacing, device)

    latent_np = extract_latent(pre_np, autoencoder, scale_factor, device)

    scheduler = define_instance(cfg, "noise_scheduler")
    scheduler.set_timesteps(1000)
    feats = add_and_denoise_one_step_ddpm(
        latent_np,
        diffusion_unet,
        mod_t,
        spacing_t,
        scheduler,
        device,
        timestep,
        top_t,
        bot_t,
    )

    out: List[torch.Tensor] = []
    for level in ("up_block_0", "up_block_1", "up_block_2", "up_block_3"):
        out.append(upsample_to_original(feats[level].squeeze(0), original_shape))

    diffusion_unet.features = {}
    del feats
    torch.cuda.empty_cache()

    return out, original_shape


def select_levels(features: Sequence[torch.Tensor], levels: str) -> List[torch.Tensor]:
    """Select feature tensors for requested level indices (string like '0123')."""
    return [features[int(ch)] for ch in levels]


def extract_source_descriptors(
    src_feats: Sequence[torch.Tensor], source_points: np.ndarray, device: torch.device
) -> torch.Tensor:
    """Build normalized per-point descriptors from selected source feature maps."""
    per_level: List[torch.Tensor] = []
    for feat in src_feats:
        f = feat[:, source_points[:, 0], source_points[:, 1], source_points[:, 2]].to(device)
        per_level.append(F.normalize(f.T[:, None], p=2, dim=-1, eps=1e-8))
    return torch.cat(per_level, dim=-1).to(device)


def flatten_target_descriptors(tgt_feats: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Normalize and flatten target voxel descriptors to (sumC, W*H*D)."""
    flat: List[torch.Tensor] = []
    for feat in tgt_feats:
        f = F.normalize(feat, p=2, dim=0, eps=1e-8)
        flat.append(einops.rearrange(f, "c w h d -> c (w h d)"))
    return torch.cat(flat, dim=0).to(device)


def best_matches_from_sims(sims: torch.Tensor, whd: Tuple[int, int, int]) -> np.ndarray:
    """Convert (N,W,H,D) sims to best-match indices (N,3) in voxel coords."""
    w, h, d = whd
    idx = sims.view(sims.shape[0], -1).argmax(dim=-1)
    coords = torch.stack(torch.unravel_index(idx, (w, h, d)), dim=1)
    return coords.cpu().numpy()


def match_points_l2reg(
    source_points: np.ndarray,
    features_source: Sequence[torch.Tensor],
    features_target: Sequence[torch.Tensor],
    levels: str,
    device: torch.device,
) -> np.ndarray:
    """Match points by cosine similarity between source point descriptors and all target voxels."""
    src_feats = select_levels(features_source, levels)
    tgt_feats = select_levels(features_target, levels)

    src_desc = extract_source_descriptors(src_feats, source_points, device)
    tgt_desc = flatten_target_descriptors(tgt_feats, device)

    sims = (torch.matmul(src_desc, tgt_desc[None]) / len(levels)).detach().cpu()
    w, h, d = tgt_feats[0].shape[1:]
    sims = sims.view(-1, w, h, d)

    matches = best_matches_from_sims(sims, (w, h, d))

    del src_desc, tgt_desc, sims
    torch.cuda.empty_cache()

    return matches


def run_job(
    job: Job,
    cfg: argparse.Namespace,
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
        feats1, _ = extract_features_all_levels(
            cfg, spacing1, job.img1, autoencoder, diffusion_unet, device, timestep, scale_factor, dataset
        )
        feats2, _ = extract_features_all_levels(
            cfg, spacing2, job.img2, autoencoder, diffusion_unet, device, timestep, scale_factor, dataset
        )

    preds: List[np.ndarray] = []
    with torch.inference_mode():
        for i in range(0, landmarks1.shape[0], batch_size):
            lm_batch = landmarks1[i : i + batch_size]
            matches = match_points_l2reg(lm_batch, feats1, feats2, levels, device)
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
    """Entry point: build jobs, load models once, run all jobs."""
    args = parse_args()
    validate_dataset_key(args.dataset)

    set_determinism(42)
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device(args.device)
    cfg = load_config(args.config)

    autoencoder, diffusion_unet, scale_factor = load_models(cfg, device)

    jobs = [make_single_job(args)]
    if not jobs:
        print("[DONE] No jobs to run.")
        return

    for j in jobs:
        print(f"[RUN] img1={j.img1.name} img2={j.img2.name} lm1={j.lm1.name}")
        run_job(
            j,
            cfg,
            autoencoder,
            diffusion_unet,
            scale_factor,
            device,
            timestep=int(args.t),
            levels=str(args.levels),
            batch_size=int(args.batch_size),
            dataset=str(args.dataset),
        )


if __name__ == "__main__":
    main()
