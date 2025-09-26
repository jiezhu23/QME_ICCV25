import numpy as np

from ..data import transform as base_transform
from ..utils.config import get_valid_args


class BaseRgbTransform:
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [0.485*255, 0.456*255, 0.406*255]
        if std is None:
            std = [0.229*255, 0.224*255, 0.225*255]
        self.mean = np.array(mean).reshape((1, 3, 1, 1))
        self.std = np.array(std).reshape((1, 3, 1, 1))

    def __call__(self, x):
        return (x - self.mean) / self.std

class BaseRgbCuttingTransform:
    def __init__(self, mean=None, std=None, cutting=None, img_w=64):
        if mean is None:
            mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        if std is None:
            std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        self.mean = np.array(mean).reshape((1, 3, 1, 1))
        self.std = np.array(std).reshape((1, 3, 1, 1))

        self.img_w = img_w
        self.cutting = cutting

    def __call__(self, x):
        """
        x: (C, H, W) or (N, C, H, W)
        output: (C, H, W) or (N, C, H, W)
        """
        if self.cutting is not None:
            cutting = self.cutting
        else:
            if x.shape[-1] == x.shape[-2]:
                cutting = x.shape[-1]//4
            elif x.shape[-1] * 2 == x.shape[-2]:
                cutting = 0
            else:
                raise ValueError
        if cutting != 0:
            x = x[..., cutting:-cutting]
        else:
            x = x
        if x.ndim == 3:
            return ((x - self.mean) / self.std).squeeze(0) # (C, H, W)
        else:
            return (x - self.mean) / self.std # (N, C, H, W)


class BasePILCuttingTransform:
    def __init__(self, mean=None, std=None, cutting=None, img_w=64):
        if mean is None:
            mean = [0.485 , 0.456 , 0.406 ]
        if std is None:
            std = [0.229 , 0.224 , 0.225 ]
        self.mean = np.array(mean).reshape((1, 3, 1, 1))
        self.std = np.array(std).reshape((1, 3, 1, 1))

        self.img_w = img_w
        self.cutting = cutting

    def __call__(self, x):
        """
        x: (C, H, W) or (N, C, H, W)
        output: (C, H, W) or (N, C, H, W)
        """
        if self.cutting is not None:
            cutting = self.cutting
        else:
            if x.shape[-1] == x.shape[-2]:
                cutting = x.shape[-1]//4
            elif x.shape[-1] * 2 == x.shape[-2]:
                cutting = 0
            else:
                raise ValueError
        if cutting != 0:
            x = x[..., cutting:-cutting]
        else:
            x = x
        if x.ndim == 3:
            return ((x - self.mean) / self.std).squeeze(0) # (C, H, W)
        else:
            return (x - self.mean) / self.std # (N, C, H, W)



def get_transform(trf_cfg=None):
    if isinstance(trf_cfg, dict):
        transform = getattr(base_transform, trf_cfg['type'])
        valid_trf_arg = get_valid_args(transform, trf_cfg, ['type'])
        return transform(**valid_trf_arg)
    if trf_cfg is None:
        return lambda x: x
    if isinstance(trf_cfg, list):
        transform = [get_transform(cfg) for cfg in trf_cfg]
        return transform
    raise "Error type for -Transform-Cfg-"
