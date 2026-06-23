#!/usr/bin/env python3
"""
Converts OULU-NPU Baseline protocol .txt files into CSV files suitable for
the OULUProtocolDataset dataloader.

CSV columns: filepath, label, label_fine, attack_type, filename
  label      : binary — 1=real, 0=attack
  label_fine : original protocol label — +1=real, -1=print, -2=replay
  attack_type: real | print1 | print2 | video_replay1 | video_replay2
"""

import os
import csv
from pathlib import Path

VIDEOS_ROOT = "/media/oem/storage01/Shakeel/facePAD_datasets/datasets/OULU_videos"
BASELINE_ROOT = "/media/oem/storage01/Shakeel/spiking_neural_net_oulu/Baseline"
CSV_OUT_ROOT = "/media/oem/storage01/Shakeel/spiking_neural_net_oulu/protocol_csvs"

FILE_TYPE_NAMES = {1: "real", 2: "print1", 3: "print2", 4: "video_replay1", 5: "video_replay2"}


def user_to_split_dir(user_id: int) -> str:
    if user_id <= 20:
        return "Train_files"
    elif user_id <= 35:
        return "Dev_files"
    else:
        return "Test_files"


def parse_txt(txt_path: Path) -> list[dict]:
    rows = []
    with open(txt_path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            label_str, filename = raw.split(",", 1)
            label_fine = int(label_str.strip())
            filename = filename.strip()

            parts = filename.split("_")
            if len(parts) != 4:
                print(f"  WARNING: unexpected filename '{filename}' in {txt_path.name}, skipping")
                continue

            user_id = int(parts[2])
            file_type = int(parts[3])
            split_dir = user_to_split_dir(user_id)
            video_path = os.path.join(VIDEOS_ROOT, split_dir, filename + ".avi")

            rows.append({
                "filepath": video_path,
                "label": 1 if label_fine == 1 else 0,
                "label_fine": label_fine,
                "attack_type": FILE_TYPE_NAMES.get(file_type, "unknown"),
                "filename": filename,
            })
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "label_fine", "attack_type", "filename"])
        writer.writeheader()
        writer.writerows(rows)


def process_protocol(protocol_dir: Path, out_dir: Path) -> None:
    print(f"\n[{protocol_dir.name}]")
    txt_files = sorted(protocol_dir.glob("*.txt"))
    for txt_path in txt_files:
        rows = parse_txt(txt_path)
        csv_name = txt_path.stem + ".csv"
        out_path = out_dir / csv_name
        write_csv(rows, out_path)
        missing = sum(1 for r in rows if not os.path.exists(r["filepath"]))
        real = sum(1 for r in rows if r["label"] == 1)
        attack = sum(1 for r in rows if r["label"] == 0)
        warn = f"  !! {missing} files not found on disk" if missing else ""
        print(f"  {csv_name}: {len(rows)} entries  (real={real}, attack={attack}){warn}")


def main():
    baseline = Path(BASELINE_ROOT)
    out_root = Path(CSV_OUT_ROOT)

    for protocol_dir in sorted(baseline.iterdir()):
        if not protocol_dir.is_dir() or not protocol_dir.name.startswith("Protocol"):
            continue
        out_dir = out_root / protocol_dir.name
        process_protocol(protocol_dir, out_dir)

    print(f"\nCSVs written to: {CSV_OUT_ROOT}")


if __name__ == "__main__":
    main()
