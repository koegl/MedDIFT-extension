
import argparse
from dataclasses import dataclass
from pathlib import Path
import re

from typing import Iterable, List, Sequence

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Job:
    """One matching job: source image, target image, source landmarks, output path."""

    img1: Path
    img2: Path
    lm1: Path
    out_csv: Path


def _sorted_unique(items: Iterable[str]) -> List[str]:
    return sorted(set(items))


class LungCTL2RegDataset(Dataset[Job]):
    """Dataset for LungCT Learn2Reg layout with *_0000/_0001 pairs."""

    def __init__(
        self,
        images_dir: Path,
        keypoints_dir: Path,
        out_dir: Path,
        case_prefix: str = "LungCT_",
        img1_suffix: str = "_0000.nii.gz",
        img2_suffix: str = "_0001.nii.gz",
        lm1_suffix: str = "_0000.csv",
        out_suffix: str = "_predicted_0001.csv",
    ) -> None:
        self.name = "LungCT_l2reg"

        self.images_dir = Path(images_dir)
        self.keypoints_dir = Path(keypoints_dir)
        self.out_dir = Path(out_dir) / self.name
        self.case_prefix = case_prefix
        self.img1_suffix = img1_suffix
        self.img2_suffix = img2_suffix
        self.lm1_suffix = lm1_suffix
        self.out_suffix = out_suffix

        self.jobs: List[Job] = self._build_jobs()

    def _build_jobs(self) -> List[Job]:
        pattern = f"{self.case_prefix}*{self.img1_suffix}"
        case_ids: List[str] = []
        for p in self.images_dir.glob(pattern):
            name = p.name
            if not name.endswith(self.img1_suffix):
                continue
            case_ids.append(name[: -len(self.img1_suffix)])

        jobs: List[Job] = []
        for case in _sorted_unique(case_ids):
            img1 = self.images_dir / f"{case}{self.img1_suffix}"
            img2 = self.images_dir / f"{case}{self.img2_suffix}"
            lm1 = self.keypoints_dir / f"{case}{self.lm1_suffix}"
            out_csv = self.out_dir / f"{case}{self.out_suffix}"

            if not img1.exists() or not img2.exists() or not lm1.exists():
                continue

            jobs.append(Job(img1=img1, img2=img2, lm1=lm1, out_csv=out_csv))

        return jobs

    def __len__(self) -> int:
        return len(self.jobs)

    def __getitem__(self, idx: int) -> Job:
        return self.jobs[idx]

    def __str__(self) -> str:
        return self.name

class LungCTDirQADataset(Dataset[Job]):
    """Dataset for LungCT-DIR-QA with caseXXXX_image1/image2 and landmarks1."""

    def __init__(
        self,
        root_dir: Path,
        out_dir: Path,
        img1_pattern: str = r"case(\d+)_image1\.nii(\.gz)?$",
        img2_template: str = "case{case}_image2.nii",
        lm1_template: str = "case{case}_landmarks1.txt",
        out_template: str = "case{case}_predicted.csv",
    ) -> None:
        self.name = "LungCT_dir_qa"

        self.root_dir = Path(root_dir)
        self.out_dir = Path(out_dir) / self.name
        self.img1_pattern = re.compile(img1_pattern)
        self.img2_template = img2_template
        self.lm1_template = lm1_template
        self.out_template = out_template

        self.jobs: List[Job] = self._build_jobs()

    def _build_jobs(self) -> List[Job]:
        case_ids: List[str] = []
        for p in self.root_dir.glob("*.nii*"):
            m = self.img1_pattern.match(p.name)
            if not m:
                continue
            case_ids.append(m.group(1).zfill(4))

        jobs: List[Job] = []
        for case in _sorted_unique(case_ids):
            img1 = self.root_dir / f"case{case}_image1.nii"
            if not img1.exists():
                img1 = self.root_dir / f"case{case}_image1.nii.gz"
            img2 = self.root_dir / self.img2_template.format(case=case)
            if not img2.exists() and img2.suffix != ".gz":
                gz_candidate = img2.with_suffix(img2.suffix + ".gz")
                if gz_candidate.exists():
                    img2 = gz_candidate
            lm1 = self.root_dir / self.lm1_template.format(case=case)
            out_csv = self.out_dir / self.out_template.format(case=case)

            if not img1.exists() or not img2.exists() or not lm1.exists():
                continue

            jobs.append(Job(img1=img1, img2=img2, lm1=lm1, out_csv=out_csv))

        return jobs

    def __len__(self) -> int:
        return len(self.jobs)

    def __getitem__(self, idx: int) -> Job:
        return self.jobs[idx]

    def __str__(self) -> str:
        return self.name

def build_dataloaders(args: argparse.Namespace) -> List[torch.utils.data.DataLoader]:
    l2reg_ds = LungCTL2RegDataset(
        images_dir=Path(args.lungct_images_dir),
        keypoints_dir=Path(args.lungct_keypoints_dir),
        out_dir=Path(args.out_dir),
    )
    dirqa_ds = LungCTDirQADataset(
        root_dir=Path(args.dirqa_root_dir),
        out_dir=Path(args.out_dir),
    )

    l2reg_loader = torch.utils.data.DataLoader(
        l2reg_ds, batch_size=1, shuffle=False, collate_fn=lambda x: x[0]
    )
    dirqa_loader = torch.utils.data.DataLoader(
        dirqa_ds, batch_size=1, shuffle=False, collate_fn=lambda x: x[0]
    )

    return [l2reg_loader, dirqa_loader]
