# MedDIFT: Multi-Scale Diffusion-Based Correspondence in 3D Medical Imaging

This repository implements **diffusion-based keypoint matching** for 3D medical images using **multi-scale diffusion features** extracted from a MAISI-trained diffusion model.  

Arxiv: https://www.arxiv.org/abs/2512.05571

## Environment Requirements

- Python ≥ 3.10  
- PyTorch ≥ 2.2  
- MONAI ≥ 1.5.0  
- SimpleITK
- NumPy
- Einops

## Installation

### 1. Clone This Repository

```
git clone https://github.com/merlinxyz/MedDIFT.git
cd MedDIFT
```

### 2. Clone the MAISI Repository (Required)
This project depends on the MAISI diffusion framework provided by MONAI.

Clone the MONAI tutorials repository and place the contents under the folder named maisi:

```
git clone https://github.com/Project-MONAI/tutorials.git
mkdir -p maisi
mv tutorials/generation/maisi/* maisi/
rm -rf tutorials
```

Run `maisi_inference_tutorial.ipynb` to download the pre-trained maisi models.

### 3. Create Conda Environment and Install Python Dependencies

### 4. Dependency Update: Custom MONAI MAISI U-Net

At present, the project relies on updating `DiffusionModelUNetMaisi`
at its installed MONAI module path using the implementation provided
in `CustomDiffusionModelUNetMaisi.py`. This approach is required to
maintain compatibility with Maisi’s configuration-based instantiation.
A cleaner and fully reproducible solution based on a forked and
customized version of MONAI will be released in a future update.

## Running 

The script supports command-line arguments for source image, target image, and source keypoint coordinates, and outputs the predicted keypoints in the target image.

```
python featurizer.py \
  --img1 /path/to/image1.nii.gz \
  --img2 /path/to/image2.nii.gz \
  --lm1  /path/to/landmarks1.csv \
  --out  predicted_lm2.csv
```
Arguments:

- `--img1`    Path to the source image
- `--img2`	Path to the target image
- `--lm1`	CSV file with source keypoints (voxel coordinates)
- `--out`	Path to save predicted keypoints in the target image
- `--config`	(Optional) MAISI configuration file (default: config_ddpm_lung)
- `--t`	(Optional) Diffusion timestep for feature extraction (default: 20)
- `--levels`	(Optional) Feature levels to use for matching (default: 0123)
- `--batch_size`	(Optional) Batch size for processing keypoints (default: 512)
- `--device`	(Optional) CUDA device (default: cuda:0)

## References

- Tang L, Jia M, Wang Q, Phoo CP, Hariharan B. Emergent correspondence from image diffusion. Advances in Neural Information Processing Systems. 2023;36:1363–89.
- Guo P, Zhao C, Yang D, Xu Z, Nath V, Tang Y et al. Maisi: Medical ai for synthetic imaging. 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE. 2025:4430–41.
- Hering A, Hansen L, Mok TCW, Heinrich MP. Learn2Reg: Comprehensive Benchmark for Image Registration. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). 2022:2087–100.