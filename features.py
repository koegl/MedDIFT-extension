from pathlib import Path
from typing import List, Sequence, Tuple

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from helper import prepare_tensors
import preprocess
from maisi.scripts.utils import define_instance


image_preprocessing_functions: Dict[str, object] = {
    "LungCT_l2reg": preprocess.preprocess_img_lungct_l2reg,
    "LungCT_dir_qa": preprocess.preprocess_img_lungct_dir_qa,
}

def load_models(
    cfg, device: torch.device
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
    cfg,
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
