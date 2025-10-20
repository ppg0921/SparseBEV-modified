import os
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

def _norm_rel(p, data_root):
    p = p.replace('\\','/')
    data_root = data_root.replace('\\','/').rstrip('/')
    if p.startswith('./'): p = p[2:]
    if p.startswith(data_root + '/'): p = p[len(data_root)+1:]
    return p

@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # If your PKL already includes per-cam 'sample_data_token', you can skip this block.
        # Otherwise, build a quick filename→token map once so we can recover tokens.
        meta = getattr(self, 'metadata', {}) or {}
        version = meta.get('version', 'v1.0-trainval')
        self._nusc = NuScenes(version=version, dataroot=self.data_root, verbose=False)

        self._cam_filename2token = {}
        for sd in self._nusc.sample_data:
            if sd['sensor_modality'] != 'camera':
                continue
            rel = sd['filename'].replace('\\','/')
            self._cam_filename2token[rel] = sd['token']
        
        if not hasattr(self, 'camera_order'):
            self.camera_order = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                                 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']
        
    def collect_sweeps(self, index, into_past=60, into_future=60):
        all_sweeps_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            all_sweeps_prev.append(self.data_infos[curr_index - 1]['cams'])
            curr_index = curr_index - 1
        
        all_sweeps_next = []
        curr_index = index + 1
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            all_sweeps_next.extend(curr_sweeps[::-1])
            all_sweeps_next.append(self.data_infos[curr_index]['cams'])
            curr_index = curr_index + 1

        return all_sweeps_prev, all_sweeps_next

    def get_data_info(self, index):
        info = self.data_infos[index]
        sweeps_prev, sweeps_next = self.collect_sweeps(index)

        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        ego2global_rotation = Quaternion(ego2global_rotation).rotation_matrix
        lidar2ego_rotation = Quaternion(lidar2ego_rotation).rotation_matrix

        input_dict = dict(
            sample_idx=info['token'],
            sweeps={'prev': sweeps_prev, 'next': sweeps_next},
            timestamp=info['timestamp'] / 1e6,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation,
        )

        if self.modality['use_camera']:
            img_paths = []
            img_timestamps = []
            lidar2img_rts = []
            cam_tokens, cam_names = [], []
            
            for cam_name in self.camera_order:
                cam_info = info['cams'][cam_name]
                
                rel_img = os.path.relpath(cam_info['data_path'])
                img_paths.append(rel_img)
                img_timestamps.append(cam_info['timestamp'] / 1e6)

                # recover / fetch sample_data token for this camera frame
                sd_token = cam_info.get('sample_data_token', None)
                if sd_token is None:
                    # fall back by filename lookup
                    rel_lookup = _norm_rel(rel_img, self.data_root)
                    sd_token = self._cam_filename2token.get(rel_lookup, None)
                    if sd_token is None:
                        # try without leading 'samples/'
                        if rel_lookup.startswith('samples/'):
                            sd_token = self._cam_filename2token.get(rel_lookup[len('samples/'):], None)
                cam_tokens.append(sd_token)
                cam_names.append(cam_name)

                # build lidar→image (4x4)
                # cam_info stores sensor2lidar (R,t) : X_lidar = R * X_cam + t
                # so lidar2cam_R = R.T ; lidar2cam_t = -R.T @ t
                R_sc = np.asarray(cam_info['sensor2lidar_rotation'])
                t_sc = np.asarray(cam_info['sensor2lidar_translation'])
                R_lc = R_sc.T
                t_lc = - R_lc @ t_sc
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T

                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)
                # lidar2cam_rt = np.eye(4, dtype=np.float32)
                # lidar2cam_rt[:3, :3] = R_lc
                # lidar2cam_rt[:3, 3]  = t_lc

                # K = np.asarray(cam_info['cam_intrinsic'])
                # viewpad = np.eye(4, dtype=np.float32)
                # viewpad[:K.shape[0], :K.shape[1]] = K

                # lidar2img_rt = viewpad @ lidar2cam_rt
                # lidar2img_rts.append(lidar2img_rt)

            # for _, cam_info in info['cams'].items():
            #     img_paths.append(os.path.relpath(cam_info['data_path']))
            #     img_timestamps.append(cam_info['timestamp'] / 1e6)

            #     # obtain lidar to image transformation matrix
            #     lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            #     lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T

            #     lidar2cam_rt = np.eye(4)
            #     lidar2cam_rt[:3, :3] = lidar2cam_r.T
            #     lidar2cam_rt[3, :3] = -lidar2cam_t
                
            #     intrinsic = cam_info['cam_intrinsic']
            #     viewpad = np.eye(4)
            #     viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            #     lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            #     lidar2img_rts.append(lidar2img_rt)

            input_dict.update(dict(
                img_filename=img_paths,
                img_timestamp=img_timestamps,
                lidar2img=lidar2img_rts,
                cam_tokens=cam_tokens,
                cam_names=cam_names
            ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict
