#!/usr/bin/env python3
"""
Organizes OULU-NPU dataset into real/attack subdirectories.

Naming convention: Phone_Session_User_File.ext
  File 1        -> real
  File 2,3,4,5  -> attack
"""

import os
import shutil
import argparse
from pathlib import Path


def organize_split(split_dir: Path, dry_run: bool = False, move: bool = False):
    real_dir = split_dir / "real"
    attack_dir = split_dir / "attack"

    if not dry_run:
        real_dir.mkdir(exist_ok=True)
        attack_dir.mkdir(exist_ok=True)

    moved, skipped = 0, 0
    for f in sorted(split_dir.iterdir()):
        if not f.is_file():
            continue

        stem = f.stem  # e.g. "1_1_01_1"
        parts = stem.split("_")
        if len(parts) != 4:
            print(f"  SKIP (unexpected name): {f.name}")
            skipped += 1
            continue

        file_type = parts[3]
        if file_type == "1":
            dest = real_dir / f.name
        elif file_type in ("2", "3", "4", "5"):
            dest = attack_dir / f.name
        else:
            print(f"  SKIP (unknown file type {file_type!r}): {f.name}")
            skipped += 1
            continue

        if dry_run:
            print(f"  {'move' if move else 'copy'}: {f.name} -> {dest.parent.name}/")
        else:
            if move:
                shutil.move(str(f), dest)
            else:
                shutil.copy2(str(f), dest)
        moved += 1

    return moved, skipped


def main():
    parser = argparse.ArgumentParser(description="Organize OULU-NPU into real/attack subfolders")
    parser.add_argument(
        "--root",
        default="/media/oem/storage01/Shakeel/facePAD_datasets/datasets/OULU_videos",
        help="Path to the OULU_videos root directory",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying (default: copy)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without touching any files",
    )
    args = parser.parse_args()

    root = Path(args.root)
    splits = {"Train_files", "Dev_files", "Test_files"}

    for split_name in sorted(splits):
        split_dir = root / split_name
        if not split_dir.is_dir():
            print(f"WARNING: {split_dir} not found, skipping.")
            continue

        print(f"\n[{split_name}]")
        moved, skipped = organize_split(split_dir, dry_run=args.dry_run, move=args.move)
        action = "Would process" if args.dry_run else ("Moved" if args.move else "Copied")
        print(f"  {action}: {moved} files  |  Skipped: {skipped}")

    print("\nDone.")


if __name__ == "__main__":
    main()
