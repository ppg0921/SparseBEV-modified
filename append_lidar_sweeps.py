import argparse
import os
import pickle
from pathlib import Path
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes


def sd_to_sweep_dict(nusc, sd_token, ref_timestamp, data_root):
    sd = nusc.get("sample_data", sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])

    rel_path = sd["filename"].replace("\\", "/")
    abs_path = os.path.join(data_root, rel_path)

    time_lag = (ref_timestamp - sd["timestamp"]) / 1e6  # seconds

    return {
        # many repos accept either key:
        "lidar_path": abs_path,
        "data_path": abs_path,

        "timestamp": sd["timestamp"],

        "lidar2ego_translation": cs["translation"],
        "lidar2ego_rotation": cs["rotation"],
        "ego2global_translation": ep["translation"],
        "ego2global_rotation": ep["rotation"],

        "time_lag": time_lag,
    }


def collect_prev_sweeps(nusc, ref_sd_token, ref_timestamp, data_root, num_prev=10):
    sweeps = []
    sd = nusc.get("sample_data", ref_sd_token)
    prev_token = sd["prev"]
    while prev_token and len(sweeps) < num_prev:
        sweeps.append(sd_to_sweep_dict(nusc, prev_token, ref_timestamp, data_root))
        sd = nusc.get("sample_data", prev_token)
        prev_token = sd["prev"]
    # order: nearest previous first
    return sweeps


def process_infos(in_pkl, out_pkl, data_root, version, num_sweeps, backfill_lidar_path=True):
    print(f"[INFO] Loading NuScenes db (version={version}, dataroot={data_root}) ...")
    nusc = NuScenes(version=version, dataroot=data_root, verbose=False)

    print(f"[INFO] Loading infos from: {in_pkl}")
    with open(in_pkl, "rb") as f:
        data = pickle.load(f)

    infos = data.get("infos", [])
    metadata = data.get("metadata", {}) or {}

    missing_token = 0
    missing_lidar_sd = 0
    updated = 0

    for i in tqdm(range(len(infos)), desc="Appending LiDAR sweeps via sample token"):
        info = infos[i]

        token = info.get("token", None)
        if token is None:
            missing_token += 1
            continue

        try:
            sample = nusc.get("sample", token)
        except KeyError:
            # this can happen if using subset JSONs or token mismatch
            missing_token += 1
            continue

        # get the reference LIDAR_TOP sample_data token for this sample
        lidar_sd_token = sample["data"].get("LIDAR_TOP", None)
        if lidar_sd_token is None:
            missing_lidar_sd += 1
            continue

        ref_sd = nusc.get("sample_data", lidar_sd_token)
        ref_ts = ref_sd["timestamp"]

        # make sure primary lidar_path is also present (helpful for loaders)
        if backfill_lidar_path and "lidar_path" not in info:
            abs_ref_path = os.path.join(data_root, ref_sd["filename"].replace("\\", "/"))
            info["lidar_path"] = abs_ref_path

        # build sweeps
        sweeps_list = collect_prev_sweeps(
            nusc, lidar_sd_token, ref_ts, data_root, num_prev=num_sweeps
        )

        info["sweeps"] = sweeps_list
        updated += 1

    # annotate metadata
    metadata = dict(metadata)
    metadata["sweeps_applied"] = f"{num_sweeps}_prev"
    # keep a hint for future readers
    suffix = "+lidar_sweeps"
    if "info_type" in metadata and suffix not in metadata["info_type"]:
        metadata["info_type"] = f"{metadata['info_type']}{suffix}"
    else:
        metadata["info_type"] = metadata.get("info_type", "generated") + suffix

    out_obj = {"infos": infos, "metadata": metadata}
    Path(out_pkl).parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Writing updated PKL to: {out_pkl}")
    with open(out_pkl, "wb") as f:
        pickle.dump(out_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    total = len(infos)
    print(f"[DONE] Updated: {updated}, Missing sample token: {missing_token}, "
          f"Missing LIDAR_TOP sd: {missing_lidar_sd}, Total: {total}")
    if updated == 0:
        print("       Hint: Ensure each info has a valid 'token' that exists in your NuScenes JSONs under dataroot.")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Append/replace multi-sweep LiDAR in NuScenes infos PKL (token-based).")
    ap.add_argument("--data_root", type=str, default='./data/nuscenes/',
                    help="NuScenes dataroot (contains samples/, sweeps/, and v1.0-*/)")
    ap.add_argument("--version", type=str, default="v1.0-trainval",
                    help="NuScenes version (e.g., v1.0-trainval, v1.0-mini)")
    ap.add_argument("--in_pkl", type=str, required=True, help="Input infos PKL (image-only ok)")
    ap.add_argument("--out_pkl", type=str, required=True, help="Output infos PKL")
    ap.add_argument("--num_sweeps", type=int, default=10, help="Number of previous LiDAR sweeps to attach")
    ap.add_argument("--no_backfill_lidar_path", action="store_true",
                    help="Do not backfill info['lidar_path'] for reference frame")
    args = ap.parse_args()

    process_infos(
        in_pkl=args.in_pkl,
        out_pkl=args.out_pkl,
        data_root=args.data_root,
        version=args.version,
        num_sweeps=args.num_sweeps,
        backfill_lidar_path=(not args.no_backfill_lidar_path),
    )


if __name__ == "__main__":
    main()
