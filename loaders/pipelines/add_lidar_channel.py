# pipelines/add_lidar_channel.py
import numpy as np
import cv2
from mmdet.datasets import PIPELINES

@PIPELINES.register_module()
class AddLidarFourthChannelFromNPZ:
    """Append a 1ch LiDAR-derived map (H,W) to each multi-view image (C=3 -> C=4).
    Assumes results['img'] shape = list of np.ndarray HxWx3 before DefaultFormatBundle3D.
    """
    def __init__(self, key='lidar4c_path_fmt', normalize=True, mean=0.0, std=1.0):
        self.key = key
        self.normalize = normalize
        self.mean = float(mean)
        self.std = float(std)

    def _load_npz(self, path):
        arr = np.load(path)['arr']  # expect key 'arr' in your npz
        # If arr has shape (H,W) or (1,H,W). If (1,H,W), squeeze.
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        return arr

    def __call__(self, results):
        imgs = results['img']  # list of HxWx3 arrays for each camera
        metas = results['img_metas']

        new_imgs = []
        new_paths = []
        for view_idx, img in enumerate(imgs):
            # Construct the npz path for this camera view.
            # You control the naming; here we use a format string provided in results:
            # e.g., results['lidar4c_path_fmt'] = '/.../{sample_token}_{cam}.npz'
            cam_name = metas['cam_names'][view_idx] if 'cam_names' in metas else f'cam{view_idx}'
            npz_path = results[self.key].format(cam=cam_name, token=metas['token'])

            m = self._load_npz(npz_path)         # H0 x W0
            H, W = img.shape[:2]
            if m.shape[0] != H or m.shape[1] != W:
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

            if self.normalize:
                # robust normalization; you can swap to global mean/std if preferred
                s = self.std if self.std > 0 else (m.std() + 1e-6)
                m = (m - self.mean) / s

            m = m.astype(img.dtype)
            m = m[..., None]                     # HxWx1
            img4 = np.concatenate([img, m], axis=2)  # HxWx4
            new_imgs.append(img4)
            new_paths.append(npz_path)

        results['img'] = new_imgs
        results['lidar4c_paths'] = new_paths
        results['img_shape'] = [im.shape for im in new_imgs]
        return results