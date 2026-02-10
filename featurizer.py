"""
Diffusion Feature-Based Keypoint Matching for Lung CT Registration

This script extracts multi-level diffusion features from a pretrained MAISI
diffusion model and uses them to perform feature-based keypoint matching
between two CT volumes. Performance is evaluated using L2 distance under
physical spacing.

Main stages:
1. Preprocess input CT volumes
2. Encode volumes into latent space using an autoencoder
3. Add noise and denoise one diffusion step while extracting UNet features
4. Upsample features to original resolution
5. Perform feature-based keypoint matching
6. Output predicted keypoint coordinates
"""

import argparse
import faulthandler
import os


faulthandler.enable(all_threads=True)  # nopep8
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # nopep8
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"  # nopep8

from typing import List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import einops
import SimpleITK as sitk

from monai.utils import set_determinism

from maisi.scripts.diff_model_setting import load_config, setup_logging
from maisi.scripts.utils import define_instance

from helper import *
from preprocess import *
# from landmark_utils import *

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
set_determinism(42)
logger = setup_logging("diffusion_feature_matching")


def extract_latent(
    image_np: np.ndarray,
    autoencoder: torch.nn.Module,
    scale_factor: torch.Tensor,
    device: torch.device = torch.device("cuda"),
) -> np.ndarray:
    """
    Encode a preprocessed 3D image into latent space.

    Args:
        image_np: Input image array [W, H, D]
        autoencoder: Trained autoencoder model
        scale_factor: Latent scaling factor from diffusion checkpoint
        device: CUDA or CPU device

    Returns:
        Latent representation as numpy array [W, H, D, C]
    """
    autoencoder.eval()

    with torch.inference_mode(), torch.amp.autocast("cuda"):
        image_tensor = (
            torch.from_numpy(image_np)
            .float()
            .to(device)
            .unsqueeze(0)
            .unsqueeze(0)  # [1, 1, W, H, D]
        )
        latent = autoencoder.encode_stage_2_inputs(image_tensor) * scale_factor
        latent = latent.float()

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Add noise at timestep t and denoise one step using DDPM-trained UNet.

    Args:
        latent_np: Latent image [W, H, D, C]
        diffusion_model: Trained diffusion UNet
        noise_scheduler: Noise scheduler
        timestep: Diffusion timestep
        modality_tensor: Modality conditioning
        spacing_tensor: Voxel spacing tensor

    Returns:
        denoised_latent: [W, H, D, C]
        predicted_velocity: [W, H, D, C]
        pred_x0: Reconstructed clean latent
        extracted_features: Dict of multi-scale UNet features
    """
    latent = (
        torch.from_numpy(latent_np)
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
        .to(device)
    )

    noise = torch.randn_like(latent)
    noisy_latent = noise_scheduler.add_noise(
        latent,
        noise,
        torch.tensor([timestep], device=device),
    )

    with torch.amp.autocast("cuda"):
        predicted_velocity = diffusion_model(
            x=noisy_latent,
            timesteps=torch.tensor([timestep], device=device),
            class_labels=modality_tensor,
            spacing_tensor=spacing_tensor,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            extract_feature=True,
        )
        extracted_features = diffusion_model.features

    pred_prev, pred_x0 = noise_scheduler.step(
        predicted_velocity, timestep, noisy_latent
    )

    def to_numpy(x):
        return x.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()

    return (
        to_numpy(pred_prev),
        to_numpy(predicted_velocity),
        to_numpy(pred_x0),
        extracted_features,
    )


def upsample_to_original(
    features: torch.Tensor,
    target_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Upsample 3D feature maps to original image resolution.

    Args:
        features: Feature tensor [C, W, H, D]
        target_shape: (W, H, D)

    Returns:
        Upsampled features [C, W, H, D]
    """
    features = features.permute(0, 3, 2, 1).unsqueeze(0).cpu().float()

    W, H, D = target_shape
    upsampled = F.interpolate(
        features,
        size=(D, H, W),
        mode="trilinear",
        align_corners=False,
    )

    return upsampled.squeeze(0).permute(0, 3, 2, 1)


def extract_features_all_levels(
    args: argparse.Namespace,
    spacing: Tuple[float, float, float],
    image_path: str,
    autoencoder,
    diffusion_unet,
    device,
    timestep: int,
    scale_factor,
    win_level: int,
    win_width: int,
) -> Tuple[List[torch.Tensor], Tuple[int, int, int], np.ndarray]:
    """
    Extract and upsample diffusion UNet features from all decoder levels.
    """
    original_np, preprocessed_np, _, original_shape = preprocess_img_l2reg(
        image_path, win_level, win_width
    )

    (
        top_region_index_tensor,
        bottom_region_index_tensor,
        spacing_tensor,
        modality_tensor,
    ) = prepare_tensors(args, spacing, device)

    latent_np = extract_latent(
        preprocessed_np, autoencoder, scale_factor, device
    )

    noise_scheduler = define_instance(args, "noise_scheduler")
    noise_scheduler.set_timesteps(1000)

    _, _, _, features = add_and_denoise_one_step_ddpm(
        latent_np,
        diffusion_unet,
        modality_tensor,
        spacing_tensor,
        noise_scheduler,
        device,
        timestep,
        top_region_index_tensor,
        bottom_region_index_tensor,
    )

    upsampled_features = []
    for level in ["up_block_0", "up_block_1", "up_block_2", "up_block_3"]:
        feat = features[level].squeeze(0)
        upsampled_features.append(
            upsample_to_original(feat, original_shape)
        )

    diffusion_unet.features = {}
    del features
    torch.cuda.empty_cache()

    return upsampled_features, original_shape, original_np


def match_points_l2reg(
    source_points: np.ndarray,
    features_source: List[torch.Tensor],
    features_target: List[torch.Tensor],
    levels: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match keypoints using cosine similarity of diffusion features.

    Args:
    source_points: (N, 3) array of keypoints in source image, each as (w, h, d) integer voxel coords
    features_source: List of feature tensors from source image at selected levels. 
                     E.g. features_source[0]: (C[0], 192, 192, 208) for C in [512,256,128,64]
    features_target: Same
    levels: String of level indices to use, e.g. "0123" to use all 4 levels
    device: CUDA or CPU device
    """
    src_feats, tgt_feats = [], []

    # 1) Select feature maps for the requested levels
    # 'map' allows you to iterate over the characters in the 'levels' string and convert them to integers
    for lvl in map(int, levels):
        src_feats.append(features_source[lvl])
        tgt_feats.append(features_target[lvl])

    # 2) Extract per-point features from the source
    # For each selected source feature map feat of shape (C, W, H, D):
    src_point_feats = []
    for feat in src_feats:
        f = feat[
            :, source_points[:, 0], source_points[:, 1], source_points[:, 2]
        ].to(device)  # this indexes all points at once, resulting in shape (C, N)

        # normalize along last dim → each point’s C-dim vector becomes unit length
        f = F.normalize(f.T[:, None], p=2, dim=-1, eps=1e-8)
        src_point_feats.append(f)

    # concatenate along feature dimension → shape (N, 1, sum(C))
    src_point_feats = torch.cat(src_point_feats, dim=-1).to(device)

    # 3) Build target descriptors for every voxel (flattened)
    tgt_feats_flat = []
    for feat in tgt_feats:
        f = F.normalize(feat, p=2, dim=0, eps=1e-8)
        tgt_feats_flat.append(
            einops.rearrange(f, "c w h d -> c (w h d)")
        )

    tgt_feats_flat = torch.cat(tgt_feats_flat, dim=0).to(device)

    # 4) Similarity computation (cosine similarity)
    sims = torch.matmul(
        src_point_feats, tgt_feats_flat.unsqueeze(0)
    ).detach().cpu() / len(levels)
    # reshape back to original spatial dimensions
    w, h, d = tgt_feats[0].shape[1:]
    sims = sims.view(-1, w, h, d)
    # sims[i, x, y, z] is now the similarity between source point i and target voxel (x,y,z)

    # 5) Pick best match per source point
    flat = sims.view(sims.shape[0], -1)
    idx = flat.argmax(dim=-1)

    # For each source point, the best-matching target coordinate
    matches = torch.stack(
        torch.unravel_index(idx, (w, h, d)), dim=1
    )

    return matches.cpu().numpy(), sims.cpu().numpy()


def calc_l2_err(pred: np.ndarray, gt: np.ndarray, spacing) -> np.ndarray:
    """
    Compute physical L2 distance under voxel spacing.
    """
    spacing = np.asarray(spacing)
    return np.linalg.norm((pred - gt) * spacing, axis=1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Diffusion feature-based keypoint matching"
    )
    parser.add_argument("--img1", type=str, required=True,
                        help="Path to source image (img1)")
    parser.add_argument("--img2", type=str, required=True,
                        help="Path to target image (img2)")
    parser.add_argument("--lm1", type=str, required=True,
                        help="Path to source keypoints CSV")
    parser.add_argument("--out", type=str, default="predicted_lm2.csv",
                        help="Output csv path for predicted keypoints")
    parser.add_argument("--config", type=str,
                        default="config_ddpm_lung", help="MAISI config name")
    parser.add_argument("--t", type=int, default=20, help="Diffusion timestep")
    parser.add_argument("--levels", type=str, default="0123",
                        help="Feature levels to use")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")

    cli_args = parser.parse_args()

    # Setup
    device = torch.device(cli_args.device)
    args = load_config(cli_args.config)

    set_determinism(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # Load inputs
    image1_path = cli_args.img1
    image2_path = cli_args.img2
    lm1_path = cli_args.lm1

    landmarks1 = np.loadtxt(lm1_path, delimiter=",")  # (N, 3)
    # landmarks1 = landmarks1[:4]  # temporary for testing

    spacing1 = sitk.ReadImage(image1_path).GetSpacing()
    spacing2 = sitk.ReadImage(image2_path).GetSpacing()

    # Load models
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    autoencoder.load_state_dict(
        torch.load(
            # r"/home/iml/fryderyk.koegl/code/MedDIFT-extension/maisi/models/autoencoder_v1.pt", map_location=device)
            args.trained_autoencoder_path, map_location=device)
    )
    autoencoder.eval()

    diffusion_unet = define_instance(args, "diffusion_unet_def").to(device)
    diffusion_ckpt = torch.load(
        # r"/home/iml/fryderyk.koegl/code/MedDIFT-extension/maisi/models/diff_unet_3d_ddpm-ct.pt", weights_only=False)
        args.trained_diffusion_path, weights_only=False)
    diffusion_unet.load_state_dict(
        diffusion_ckpt["unet_state_dict"], strict=False
    )
    diffusion_unet.eval()

    scale_factor = diffusion_ckpt["scale_factor"].to(device)

    # Feature extraction
    win_level, win_width = 0, 2000
    t = cli_args.t

    print(f"[INFO] Extracting diffusion features at t={t}")

    with torch.no_grad():
        features1, original_shape1, _ = extract_features_all_levels(
            args, spacing1, image1_path,
            autoencoder, diffusion_unet,
            device, t, scale_factor,
            win_level, win_width
        )

        features2, original_shape2, _ = extract_features_all_levels(
            args, spacing2, image2_path,
            autoencoder, diffusion_unet,
            device, t, scale_factor,
            win_level, win_width
        )

    # Landmark matching
    level_str = cli_args.levels
    batch_size = cli_args.batch_size

    all_matches = []

    print(f"[INFO] Matching keypoints using levels {level_str}")

    with torch.inference_mode():
        for i in range(0, landmarks1.shape[0], batch_size):
            lm_batch = landmarks1[i: i + batch_size]

            print(
                f"  Processing batch {i // batch_size + 1} / {(landmarks1.shape[0] + batch_size - 1) // batch_size}")

            matches, sims = match_points_l2reg(
                lm_batch,
                features1,
                features2,
                level_str,
                device,
            )

            all_matches.append(matches)

            del matches, sims, lm_batch
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    predicted_lm2 = np.vstack(all_matches)

    os.makedirs(os.path.dirname(cli_args.out) or ".", exist_ok=True)
    np.savetxt(cli_args.out, predicted_lm2, delimiter=",", fmt="%.2f")

    print(f"[DONE] Predicted keypoints saved to: {cli_args.out}")
