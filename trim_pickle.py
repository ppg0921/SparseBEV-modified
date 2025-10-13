#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter a NuScenes infos pickle to include only samples whose files exist locally.

- Keeps annotation fields (valid_flag, gt_boxes, etc.) from the original PKL.
- For each sample:
  * Checks the 6 camera image files exist; if --require-all-6 is set (default),
    the sample is kept only if all 6 are present.
    Otherwise, keeps samples that have >= --min-cams (default 1) and prunes the 'cams' dict.
  * Prunes sweeps: keeps only sweep frames whose files exist; drops empty sweeps.
- Optionally normalizes 'data_path' to point at your --data-root if the PKL used a different prefix.

Usage:
  python filter_infos_to_local_files.py \
    --in-pkl ./data/nuscenes/nuscenes_infos_train.pkl \
    --out-pkl ./data/nuscenes/nuscenes_infos_train_local.pkl \
    --data-root ./data/nuscenes \
    --require-all-6   # default True; remove to allow partial cams
"""

import os
import argparse
import mmcv

CAM_TYPES = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
]

def normalize_path(p: str, data_root: str) -> str:
    """Try to map a PKL path to your local data_root if needed."""
    if os.path.exists(p):
        return p
    # If PKL stored 'data/nuscenes/...' but your local root is different:
    needle = "data/nuscenes/"
    if needle in p:
        suffix = p.split(needle, 1)[1]  # e.g. 'samples/CAM_FRONT/xxx.jpg'
        candidate = os.path.join(data_root, suffix)
        if os.path.exists(candidate):
            return candidate
    # If PKL stored only 'samples/...':
    if p.startswith("samples/") or p.startswith("sweeps/") or p.startswith("maps/"):
        candidate = os.path.join(data_root, p)
        if os.path.exists(candidate):
            return candidate
    return p  # return as-is; existence will be checked by caller

def file_exists(p: str) -> bool:
    try:
        return os.path.exists(p)
    except Exception:
        return False

def filter_sample(info, data_root, min_cams, require_all_6):
    """Return a pruned copy of `info` or None if it should be dropped."""
    info = dict(info)  # shallow copy

    # ---- Filter keyframe cams ----
    cams = info.get("cams", {})
    pruned_cams = {}
    for cam in CAM_TYPES:
        camd = cams.get(cam)
        if not isinstance(camd, dict):
            continue
        dp = camd.get("data_path")
        if not isinstance(dp, str):
            continue
        new_path = normalize_path(dp, data_root)
        if file_exists(new_path):
            camd = dict(camd)
            camd["data_path"] = new_path
            pruned_cams[cam] = camd
    # Decide keep/drop based on cams
    if require_all_6:
        if len(pruned_cams) != 6:
            return None
    else:
        if len(pruned_cams) < min_cams:
            return None
    info["cams"] = pruned_cams

    # ---- Filter sweeps (list of dicts of cams) ----
    sweeps = info.get("sweeps", [])
    pruned_sweeps = []
    for sweep in sweeps or []:
        keep_cams = {}
        for cam, camd in sweep.items():
            if not isinstance(camd, dict):
                continue
            dp = camd.get("data_path")
            if not isinstance(dp, str):
                continue
            new_path = normalize_path(dp, data_root)
            if file_exists(new_path):
                camd = dict(camd)
                camd["data_path"] = new_path
                keep_cams[cam] = camd
        if keep_cams:
            pruned_sweeps.append(keep_cams)
    info["sweeps"] = pruned_sweeps

    # ---- (Optional) lidar_path relink if present ----
    if "lidar_path" in info and isinstance(info["lidar_path"], str):
        lp = normalize_path(info["lidar_path"], data_root)
        if file_exists(lp):
            info["lidar_path"] = lp
        else:
            # Typically not fatal for image-only pipelines; comment out to drop sample instead:
            # return None
            pass

    return info

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-pkl", required=True)
    ap.add_argument("--out-pkl", required=True)
    ap.add_argument("--data-root", default="./data/nuscenes/", help="Path to your local nuScenes root (contains samples/, sweeps/)")
    ap.add_argument("--require-all-6", action="store_true", default=True,
                    help="Keep only samples that have ALL 6 camera images. (default True)")
    ap.add_argument("--no-require-all-6", action="store_false", dest="require_all_6",
                    help="Allow partial cameras; use --min-cams to set minimum.")
    ap.add_argument("--min-cams", type=int, default=1, help="Minimum number of cameras if not requiring all 6.")
    args = ap.parse_args()

    d = mmcv.load(args.in_pkl)
    infos = d.get("infos", [])
    meta = d.get("metadata", {})
    total = len(infos)
    kept, dropped = 0, 0
    out_infos = []

    for info in infos:
        pruned = filter_sample(info, args.data_root, args.min_cams, args.require_all_6)
        if pruned is None:
            dropped += 1
        else:
            kept += 1
            out_infos.append(pruned)

    out = {
        "infos": out_infos,
        "metadata": dict(meta),
    }
    # Tag output so you can identify it later
    out["metadata"]["info_type"] = "filtered_local_files"
    out["metadata"]["dataset"] = out["metadata"].get("dataset", "nuscenes")

    mmcv.dump(out, args.out_pkl)
    print(f"[filter] input: {args.in_pkl}")
    print(f"[filter] data_root: {args.data_root}")
    print(f"[filter] total: {total} | kept: {kept} | dropped: {dropped}")
    print(f"[filter] wrote: {args.out_pkl}")

if __name__ == "__main__":
    main()
