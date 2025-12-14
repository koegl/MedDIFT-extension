import argparse
import json
import torch
import numpy as np

def load_config(config_path: str) -> argparse.Namespace:
    """
    Load configuration from JSON files.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        argparse.Namespace: Loaded configuration.
    """
    args = argparse.Namespace()

    with open(config_path, "r") as f:
        model_config = json.load(f)
    for k, v in model_config.items():
        setattr(args, k, v)

    return args

def transfer_lm_simple(lm1, img1_shape, img2_shape):
    return lm1 * (img1_shape / img2_shape)

def prepare_tensors(args: argparse.Namespace, spacing: tuple, device: torch.device) -> tuple:
    """
    Prepare necessary tensors for inference.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to load tensors on.

    Returns:
        tuple: Prepared top_region_index_tensor, bottom_region_index_tensor, and spacing_tensor.
    """
    top_region_index_tensor = np.array(args.diffusion_unet_inference["top_region_index"]).astype(float) * 1e2
    bottom_region_index_tensor = np.array(args.diffusion_unet_inference["bottom_region_index"]).astype(float) * 1e2
    spacing_tensor = np.array(spacing).astype(float) * 1e2

    top_region_index_tensor = torch.from_numpy(top_region_index_tensor[np.newaxis, :]).half().to(device)
    bottom_region_index_tensor = torch.from_numpy(bottom_region_index_tensor[np.newaxis, :]).half().to(device)
    spacing_tensor = torch.from_numpy(spacing_tensor[np.newaxis, :]).half().to(device)
    modality_tensor = args.diffusion_unet_inference["modality"] * torch.ones(
        (len(spacing_tensor)), dtype=torch.long
    ).to(device)

    return top_region_index_tensor, bottom_region_index_tensor, spacing_tensor, modality_tensor