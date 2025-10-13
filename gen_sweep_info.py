# Generate info files manually
import os
import mmcv
import tqdm
import pickle
import argparse
import numpy as np
from nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.splits import create_splits_scenes


parser = argparse.ArgumentParser()
parser.add_argument('--data-root', default='data/nuscenes')
parser.add_argument('--version', default='v1.0-trainval')
args = parser.parse_args()

CAM_TYPES = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
]

def to_rotmat(x):
    """x may be 3x3 or a 4-list quaternion [w,x,y,z]. Return 3x3 rotation matrix."""
    arr = np.array(x, dtype=float)
    if arr.shape == (3,3):
        return arr
    if arr.size == 4:
        return Quaternion(arr.tolist()).rotation_matrix
    raise ValueError(f"Unexpected rotation shape: {arr.shape}")


def compose_sensor2lidar(cam_R_s2g, cam_t_s2g, lidar_R_l2g, lidar_t_l2g):
    R_s2g = to_rotmat(cam_R_s2g)
    t_s2g = np.array(cam_t_s2g, dtype=float)
    R_l2g = to_rotmat(lidar_R_l2g)
    t_l2g = np.array(lidar_t_l2g, dtype=float)
    R_s2l = R_l2g.T @ R_s2g
    t_s2l = R_l2g.T @ (t_s2g - t_l2g)
    return R_s2l, t_s2l


def compose_q_e2g_s2e_to_s2g(q_e2g_list, q_s2e_list):
    q_e2g = Quaternion(q_e2g_list)   # [w,x,y,z]
    q_s2e = Quaternion(q_s2e_list)
    q_s2g = q_e2g * q_s2e            # sensor->global
    return [q_s2g.w, q_s2g.x, q_s2g.y, q_s2g.z]

def rotate_by_quat(q_list, v):
    """Rotate vector v by quaternion q (list [w,x,y,z])."""
    return Quaternion(q_list).rotate(np.array(v, dtype=float))

def quat_to_mat(q):
    """[w,x,y,z] -> 3x3 rotation matrix"""
    return Quaternion(q).rotation_matrix

import numpy as np
from pyquaternion import Quaternion

def quat_to_mat(q):
    """[w,x,y,z] -> 3x3 rotation matrix"""
    return Quaternion(q).rotation_matrix

def cam_to_lidar_from_cam_and_lidar_pose(q_cam2ego, t_cam2ego, q_lidar2ego, t_lidar2ego):
    """
    Return R_cl, t_cl such that X_lidar = R_cl @ X_cam + t_cl
    Using: cam->ego (R_ce,t_ce) and lidar->ego (R_le,t_le)
    cam->lidar = (ego->lidar).  (cam->ego)
    """
    R_ce = quat_to_mat(q_cam2ego)
    t_ce = np.array(t_cam2ego, dtype=float)

    R_le = quat_to_mat(q_lidar2ego)
    t_le = np.array(t_lidar2ego, dtype=float)

    # ego->lidar
    R_el = R_le.T
    t_el = - R_el @ t_le

    # cam->lidar
    R_cl = R_el @ R_ce
    t_cl = R_el @ t_ce + t_el
    return R_cl, t_cl

def get_cam_info(nusc, sample_data, lidar_q_s2e, lidar_t_s2e):
    data_path = os.path.join(args.data_root, sample_data['filename'])
    if not os.path.exists(data_path):
        return None

    pose_record = nusc.get('ego_pose', sample_data['ego_pose_token'])               # ego pose at cam timestamp
    cs_record   = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])

    q_cam2ego = cs_record['rotation']           # quaternion [w,x,y,z]
    t_cam2ego = cs_record['translation']        # (3,)
    q_ego2glob = pose_record['rotation']        # we won't store this for cams
    t_ego2glob = pose_record['translation']

    # sensor->global (matrix + vector) - optional (not required by your loader)
    R_cam2ego = quat_to_mat(q_cam2ego)
    R_ego2glob = quat_to_mat(q_ego2glob)
    R_cam2glob = R_cam2ego.T @ R_ego2glob.T
    t_cam2glob = np.array(t_cam2ego) @ R_ego2glob.T + np.array(t_ego2glob)

    # REQUIRED by your loader: camera -> lidar (matrix + vector)
    R_cl, t_cl = cam_to_lidar_from_cam_and_lidar_pose(
        q_cam2ego, t_cam2ego, lidar_q_s2e, lidar_t_s2e
    )

    cam_intrinsic = np.array(cs_record['camera_intrinsic'])

    return {
        'data_path': data_path,
        'timestamp': sample_data['timestamp'],
        'cam_intrinsic': cam_intrinsic,

        # what your loader needs:
        'sensor2lidar_rotation': R_cl,
        'sensor2lidar_translation': t_cl,

        # optional (keep if useful elsewhere):
        'sensor2global_rotation': R_cam2glob,
        'sensor2global_translation': t_cam2glob,
    }

def add_sweep_info(nusc, sample_infos, max_sweeps=5):
    for curr_id in tqdm.tqdm(range(len(sample_infos['infos']))):
        base = sample_infos['infos'][curr_id]
        sample = nusc.get('sample', base['token'])

        cam_types = CAM_TYPES
        # keyframe cam_data (refresh with sensor2lidar)
        curr_cams = {cam: nusc.get('sample_data', sample['data'][cam]) for cam in cam_types}

        # ensure keyframe cams written correctly (idempotent)
        for cam in cam_types:
            sd = curr_cams[cam]
            ci = get_cam_info(nusc, sd, base['lidar2ego_rotation'], base['lidar2ego_translation'])
            if ci is not None:
                base['cams'].setdefault(cam, {})
                base['cams'][cam].update(ci)
            else:
                base['cams'].pop(cam, None)

        # collect sweeps (prev) for each cam
        sweep_infos = []
        for _ in range(max_sweeps):
            sweep, advanced = {}, False
            for cam in cam_types:
                prev_tok = curr_cams[cam]['prev']
                if prev_tok == '':
                    continue
                prev_sd = nusc.get('sample_data', prev_tok)
                ci = get_cam_info(nusc, prev_sd, base['lidar2ego_rotation'], base['lidar2ego_translation'])
                curr_cams[cam] = prev_sd
                advanced = True
                if ci is not None:
                    sweep[cam] = ci
            if sweep:
                sweep_infos.append(sweep)
            if not advanced:
                break

        base['sweeps'] = sweep_infos
    return sample_infos


def get_ego_global_pose_from_lidar(nusc, sample):
    """Use LIDAR_TOP ego pose as canonical ego->global for the sample."""
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    ego_t = np.array(pose['translation'], dtype=float)
    ego_R = Quaternion(pose['rotation']).rotation_matrix
    return ego_t, ego_R


def build_base_infos(nusc, dataroot, split_name):
    """
    Build a base sample_infos using the ORIGINAL get_cam_info for keyframes:
      - cams dict populated with per-cam info (skipping missing files)
      - top-level ego pose from LIDAR_TOP
      - metadata/version included
    """
    # which scenes belong to this split
    from nuscenes.utils.splits import create_splits_scenes
    split_scenes = create_splits_scenes()[split_name]
    name2tok = {s['name']: s['token'] for s in nusc.scene}
    target_scene_tokens = {name2tok[name] for name in split_scenes if name in name2tok}

    sample_infos = {
        "infos": [],
        "metadata": {"version": args.version, "dataset": "nuscenes", "info_type": "sparsebev_base"},
    }

    for sample in nusc.sample:
        if sample['scene_token'] not in target_scene_tokens:
            continue

        # top-level ego & lidar (store quats for ego/lidar rotations)
        lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
        ego_t = np.array(ego_pose['translation'], dtype=float)
        ego_q = ego_pose['rotation']  # [w,x,y,z]

        cs_lidar = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
        lidar_t_s2e = np.array(cs_lidar['translation'], dtype=float)
        lidar_q_s2e = cs_lidar['rotation']  # [w,x,y,z]
        lidar_path = os.path.join(dataroot, lidar_sd['filename'])

        # compose lidar->global (quaternion + vector)
        # q_lidar2glob = q_ego * q_lidar2ego
        q_lidar2glob = (Quaternion(ego_q) * Quaternion(lidar_q_s2e))
        lidar_q_s2g = [q_lidar2glob.w, q_lidar2glob.x, q_lidar2glob.y, q_lidar2glob.z]
        lidar_t_s2g = Quaternion(ego_q).rotate(lidar_t_s2e) + ego_t

        # Build cams; now we can compute cam->lidar using (lidar_q_s2e, lidar_t_s2e)
        cams = {}
        for cam in CAM_TYPES:
            sd = nusc.get('sample_data', sample['data'][cam])
            ci = get_cam_info(nusc, sd, lidar_q_s2e, lidar_t_s2e)
            if ci is not None:
                cams[cam] = ci
        if not cams:
            continue

        info = {
            "token": sample["token"],
            "timestamp": sample["timestamp"],
            "scene_name": nusc.get('scene', sample['scene_token'])['name'],

            # top-level (as QUATs where rotations are expected by your loader)
            "ego2global_translation": ego_t,
            "ego2global_rotation": ego_q,               # quaternion (loader expects this)
            "lidar_path": lidar_path,
            "lidar2ego_translation": lidar_t_s2e,
            "lidar2ego_rotation": lidar_q_s2e,          # quaternion (loader expects this)
            "lidar2global_translation": lidar_t_s2g,
            "lidar2global_rotation": lidar_q_s2g,       # quaternion

            "cams": cams,
        }
        sample_infos["infos"].append(info)

    return sample_infos

def version_to_splits(version):
    """
    Map NuScenes version string to create_splits_scenes() keys and default filenames.
    """
    if version == 'v1.0-trainval':
        return [
            ('train', 'nuscenes_infos_train_sweep.pkl'),
            ('val',   'nuscenes_infos_val_sweep.pkl'),
        ]
    elif version == 'v1.0-test':
        return [
            ('test',  'nuscenes_infos_test_sweep.pkl'),
        ]
    elif version == 'v1.0-mini':
        return [
            ('mini_train', 'nuscenes_infos_train_mini_sweep.pkl'),
            ('mini_val',   'nuscenes_infos_val_mini_sweep.pkl'),
        ]
    else:
        raise ValueError(f'Unknown version: {version}')

# if __name__ == '__main__':
#     nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=True)
#     print("preparing base pickle files...")
#     for split_name, out_name in version_to_splits(args.version):
#         print(f'Building base infos for split "{split_name}" ...')
#         sample_infos = build_base_infos(nusc, args.data_root, split_name)

#         print(f'Adding sweeps (skipping missing files) for split "{split_name}" ...')
#         sample_infos = add_sweep_info(nusc, sample_infos)

#         # --- ensure metadata present ---
#         md = sample_infos.setdefault("metadata", {})
#         md.setdefault("version", args.version)
#         md.setdefault("dataset", "nuscenes")
#         # optional provenance tag:
#         md.setdefault("info_type", "sparsebev_sweep")

#         out_path = os.path.join(args.data_root, out_name)
#         mmcv.dump(sample_infos, out_path)
#         print(f'Wrote {len(sample_infos["infos"])} samples -> {out_path}')
        
        
if __name__ == '__main__':
    nusc = NuScenes(args.version, args.data_root)

    if args.version == 'v1.0-trainval':
        sample_infos = pickle.load(open(os.path.join(args.data_root, 'nuscenes_infos_train.pkl'), 'rb'))
        sample_infos = add_sweep_info(nusc, sample_infos)
        mmcv.dump(sample_infos, os.path.join(args.data_root, 'nuscenes_infos_train_sweep.pkl'))

        sample_infos = pickle.load(open(os.path.join(args.data_root, 'nuscenes_infos_val.pkl'), 'rb'))
        sample_infos = add_sweep_info(nusc, sample_infos)
        mmcv.dump(sample_infos, os.path.join(args.data_root, 'nuscenes_infos_val_sweep.pkl'))

    elif args.version == 'v1.0-test':
        sample_infos = pickle.load(open(os.path.join(args.data_root, 'nuscenes_infos_test.pkl'), 'rb'))
        sample_infos = add_sweep_info(nusc, sample_infos)
        mmcv.dump(sample_infos, os.path.join(args.data_root, 'nuscenes_infos_test_sweep.pkl'))

    elif args.version == 'v1.0-mini':
        sample_infos = pickle.load(open(os.path.join(args.data_root, 'nuscenes_infos_train_mini.pkl'), 'rb'))
        sample_infos = add_sweep_info(nusc, sample_infos)
        mmcv.dump(sample_infos, os.path.join(args.data_root, 'nuscenes_infos_train_mini_sweep.pkl'))

        sample_infos = pickle.load(open(os.path.join(args.data_root, 'nuscenes_infos_val_mini.pkl'), 'rb'))
        sample_infos = add_sweep_info(nusc, sample_infos)
        mmcv.dump(sample_infos, os.path.join(args.data_root, 'nuscenes_infos_val_mini_sweep.pkl'))

    else:
        raise ValueError
