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
    def __init__(self, npz_key=None, normalize=True, mean=0.0, std=1.0):
        """
        npz_key: if None, auto-pick the only array in the archive or common names ['depth','arr','map'].
        """
        self.npz_key = npz_key
        self.normalize = normalize
        self.mean = float(mean)
        self.std = float(std)

    def _read_npz(self, p):
        d = np.load(p)
        if self.npz_key is not None:
            if self.npz_key not in d.files:
                raise KeyError(f"{self.npz_key} not found in {p}; available: {d.files}")
            a = d[self.npz_key]
        else:
            # auto-pick
            if len(d.files) == 1:
                a = d[d.files[0]]
            else:
                for k in ('depth', 'arr', 'map', 'channel'):
                    if k in d.files:
                        a = d[k]
                        break
                else:
                    raise KeyError(f"No npz_key specified and could not infer from keys {d.files} in {p}")
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        return a

    def __call__(self, results):
        imgs = results['img']                      # list of images (maybe V or V*T)
        cam_tokens = results.get('cam_tokens', None)
        if cam_tokens is None:
            raise KeyError("cam_tokens missing; make sure dataset emits it.")
        n_views = len(cam_tokens)
        if n_views == 0:
            raise ValueError("cam_tokens is empty.")

        base = results.get('fourth_root', '')
        total = len(imgs)

        # Allow temporal stacking: imgs can be [V] or [V*T]
        if total % n_views != 0:
            raise ValueError(f"num imgs ({total}) not a multiple of num views ({n_views}).")

        new_imgs, paths = [], []
        for i, img in enumerate(imgs):
            view_idx = i % n_views     # ← key change: wrap per view
            tok = cam_tokens[view_idx]
            npz_path = f"{base}/upsampled/{tok}.npz"
            m = self._read_npz(npz_path)

            H, W = img.shape[:2]
            if m.shape[:2] != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

            if self.normalize:
                s = self.std if self.std > 0 else (m.std() + 1e-6)
                m = (m - self.mean) / s

            m = m.astype(img.dtype)[..., None]      # HxWx1
            img4 = np.concatenate([img, m], axis=2) # HxWx4
            new_imgs.append(img4)
            paths.append(npz_path)

        results['img'] = new_imgs
        results['fourth_paths'] = paths
        results['img_shape'] = [im.shape for im in new_imgs]
        return results

