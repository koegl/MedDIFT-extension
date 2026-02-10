from __future__ import annotations

import re
from pathlib import Path


def rename_cases_zfill(width: int = 4, dry_run: bool = True) -> None:
    """Rename files like case10_image1.nii -> case0010_image1.nii."""
    root = Path("/home/iml/fryderyk.koegl/data/Lung-DIR-QA/nii-txt")

    pattern = re.compile(r"^(case)(\d+)(_.+)$")

    renames: list[tuple[Path, Path]] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue

        m = pattern.match(p.name)
        if not m:
            continue

        prefix, num_str, suffix = m.groups()
        new_name = f"{prefix}{int(num_str):0{width}d}{suffix}"
        new_path = p.with_name(new_name)

        if new_path == p:
            continue

        if new_path.exists():
            raise FileExistsError(f"Target already exists: {new_path}")

        renames.append((p, new_path))

    for old, new in renames:
        print(f"{old.name}  ->  {new.name}")
        if not dry_run:
            old.rename(new)


if __name__ == "__main__":
    rename_cases_zfill(width=4, dry_run=False)
