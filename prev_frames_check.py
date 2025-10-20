# diag_sweeps.py
import os, argparse
from nuscenes import NuScenes

CAM_TYPES = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT']

ap = argparse.ArgumentParser()
ap.add_argument('--data-root', default="./data/nuscenes/")
ap.add_argument('--version', default="v1.0-trainval")
ap.add_argument('--limit', type=int, default=20)
args = ap.parse_args()

nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)

def exists(sd):
    p = os.path.join(args.data_root, sd['filename'])
    return os.path.exists(p), p

cnt_prev_token, cnt_prev_file = 0, 0
for i, sample in enumerate(nusc.sample):
    if i >= args.limit: break
    print(f"\nSample {i} token={sample['token']} scene_token={sample['scene_token']} key_prev={sample['prev']!r}")
    for cam in CAM_TYPES:
        sd = nusc.get('sample_data', sample['data'][cam])
        has_prev = (sd['prev'] != '')
        msg = f"  {cam}: is_key_frame={sd['is_key_frame']} prev_token={'YES' if has_prev else 'NO'}"
        if has_prev:
            cnt_prev_token += 1
            prev_sd = nusc.get('sample_data', sd['prev'])
            ok, path = exists(prev_sd)
            msg += f" | prev_file={'OK' if ok else 'MISSING'} -> {path}"
            if ok: cnt_prev_file += 1
        print(msg)

print(f"\nSummary: cams with prev tokens: {cnt_prev_token}, with prev files on disk: {cnt_prev_file}")
