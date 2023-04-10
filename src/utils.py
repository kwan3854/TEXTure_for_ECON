import os
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from matplotlib import cm
from PIL import Image


def get_view_direction(thetas, phis, overhead, front):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis

    # res[(phis < front)] = 0
    res[(phis >= (2 * np.pi - front / 2)) & (phis < front / 2)] = 0

    # res[(phis >= front) & (phis < np.pi)] = 1
    res[(phis >= front / 2) & (phis < (np.pi - front / 2))] = 1

    # res[(phis >= np.pi) & (phis < (np.pi + front))] = 2
    res[(phis >= (np.pi - front / 2)) & (phis < (np.pi + front / 2))] = 2

    # res[(phis >= (np.pi + front))] = 3
    res[(phis >= (np.pi + front / 2)) & (phis < (2 * np.pi - front / 2))] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res


def tensor2numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return tensor


def make_path(path: Path) -> Path:
    path.mkdir(exist_ok=True, parents=True)
    return path


def save_colormap(tensor: torch.Tensor, path: Path):
    Image.fromarray((cm.seismic(tensor.cpu().numpy())[:, :, :3] * 255).astype(np.uint8)).save(path)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def smooth_image(self, img: torch.Tensor, sigma: float) -> torch.Tensor:
    """apply gaussian blur to an image tensor with shape [C, H, W]"""
    img = T.GaussianBlur(kernel_size=(51, 51), sigma=(sigma, sigma))(img)
    return img


def get_nonzero_region(mask: torch.Tensor):
    # Get the indices of the non-zero elements
    nz_indices = mask.nonzero()
    # Get the minimum and maximum indices along each dimension
    min_h, max_h = nz_indices[:, 0].min(), nz_indices[:, 0].max()
    min_w, max_w = nz_indices[:, 1].min(), nz_indices[:, 1].max()

    # Calculate the size of the square region
    size = max(max_h - min_h + 1, max_w - min_w + 1) * 1.01
    # Calculate the upper left corner of the square region
    h_start = min(min_h, max_h) - (size - (max_h - min_h + 1)) / 2
    w_start = min(min_w, max_w) - (size - (max_w - min_w + 1)) / 2

    min_h = int(h_start)
    min_w = int(w_start)
    max_h = int(min_h + size)
    max_w = int(min_w + size)

    return [min_h, min_w, max_h, max_w]


def crop_or_pad_image(img, crop_info):

    [min_h, min_w, max_h, max_w] = crop_info
    # size = img.shape[-1]
    # dim = img.shape[1]

    # if min_h < 0 or min_w < 0:
    #     # Calculate the padding needed to make the output square
    #     pad = int(abs(min(min_h, min_w)))
    #     if min_w < 0:
    #         min_w = 0

    #     if min_h < 0:
    #         min_h = 0

    #     # Create a new canvas with the padded dimensions and fill it with black color
    #     canvas = torch.zeros((img.shape[0], dim, size, size)).type_as(img)
    #     # Copy the region from the original image to the canvas
    #     canvas[:, :, pad + min_h:pad + max_h, pad + min_w:pad + max_w] = img[:, :, min_h:max_h,
    #                                                                          min_w:max_w]

    #     # Return the padded image
    #     return canvas

    # else:
    #     # If the crop region is valid, just crop the image using OpenCV's slicing syntax
    return img[:, :, min_h:max_h, min_w:max_w]


def gaussian_fn(M, std):
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n**2 / sig2)
    return w


def gkern(kernlen=256, std=128):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian_fn(kernlen, std=std)
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d


def gaussian_blur(image: torch.Tensor, kernel_size: int, std: int) -> torch.Tensor:
    gaussian_filter = gkern(kernel_size, std=std)
    gaussian_filter /= gaussian_filter.sum()

    image = F.conv2d(
        image, gaussian_filter.unsqueeze(0).unsqueeze(0).cuda(), padding=kernel_size // 2
    )
    return image


def color_with_shade(color: List[float], z_normals: torch.Tensor, light_coef=0.7):
    normals_with_light = (light_coef + (1 - light_coef) * z_normals.detach())
    shaded_color = torch.tensor(color).view(1, 3, 1, 1).to(z_normals.device) * normals_with_light
    return shaded_color
