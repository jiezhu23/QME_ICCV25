import numpy as np
from PIL import Image
import torch

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
        # If input is a PIL image, convert to NumPy array
        if isinstance(x, Image.Image):
            x = np.array(x).astype(np.float32) / 255.0  # Convert to NumPy array and normalize to [0, 1]
            x = np.transpose(x, (2, 0, 1))  # Convert (H, W, C) to (C, H, W)
            
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
            return torch.tensor(((x - self.mean) / self.std).squeeze(0)).float() # (C, H, W)
        else:
            return torch.tensor((x - self.mean) / self.std).float() # (N, C, H, W)

