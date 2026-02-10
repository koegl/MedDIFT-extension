from typing import List, Sequence, Tuple

import einops
import numpy as np
import torch
import torch.nn.functional as F

from features import select_levels


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
