# projects/pipelines/fourth_channel.py
import os
import numpy as np
import cv2
from mmdet.datasets import PIPELINES

@PIPELINES.register_module()
class SetFourthChanRoot:
    """Stores dataroot for later path resolution."""
    def __init__(self, dataroot):
        self.dataroot = dataroot
    def __call__(self, results):
        results['fourth_root'] = self.dataroot
        return results

@PIPELINES.register_module()
class AddFourthChannelFromNPZ:
    """
    Append a 1-channel LiDAR-derived map (H,W) loaded from:
        {fourth_root}/upsampled/{cam_token}.npz
    The npz must contain array under key 'arr'.
    Assumes results contain per-view cam tokens in results['cam_tokens'].
    """
    def __init__(self, npz_key='arr', normalize=True, mean=0.0, std=1.0):
        self.npz_key = npz_key
        self.normalize = normalize
        self.mean = float(mean)
        self.std = float(std)

    def _read_npz(self, p):
        d = np.load(p)
        a = d[self.npz_key]
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        return a

    def __call__(self, results):
        assert 'img' in results, "Images must be loaded before AddFourthChannelFromNPZ."
        imgs = results['img']                 # list of HxWx3 np.ndarrays (per view)
        cam_tokens = results.get('cam_tokens', None)
        assert cam_tokens is not None, "cam_tokens missing; ensure dataset fills cam tokens."
        base = results.get('fourth_root', '')

        new_imgs = []
        paths = []
        for view_idx, img in enumerate(imgs):
            tok = cam_tokens[view_idx]
            npz_path = os.path.join(base, 'upsampled', f'{tok}.npz')
            m = self._read_npz(npz_path)      # H0 x W0

            H, W = img.shape[:2]
            if m.shape[:2] != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

            if self.normalize:
                s = self.std if self.std > 0 else (m.std() + 1e-6)
                m = (m - self.mean) / s

            m = m.astype(img.dtype)[..., None]     # HxWx1
            img4 = np.concatenate([img, m], axis=2)  # HxWx4
            new_imgs.append(img4)
            paths.append(npz_path)

        results['img'] = new_imgs
        results['fourth_paths'] = paths
        # keep shapes consistent for later transforms
        results['img_shape'] = [im.shape for im in new_imgs]
        return results
