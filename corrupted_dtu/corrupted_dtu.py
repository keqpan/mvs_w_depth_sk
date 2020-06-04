from pathlib import Path

import cv2
from PIL import Image
import imageio
import numpy as np
import torch


def read_rgb(rgb_root, scan, cam_id, light_id, train):
    scan = scan + ('_train' if train else '')
    rgb = rgb_root / scan
    rgb = rgb / f'rect_{cam_id + 1:03}_{light_id}_r5000.png'
    rgb = Image.open(rgb)
    rgb = np.asarray(rgb)
    rgb = torch.from_numpy(rgb)
    return rgb


def read_depth(depth_root, scan, cam_id, train):
    d = depth_root / scan
    d = d / f'depth_map_{cam_id:04}.pfm'
    d = imageio.imread(d)[::-1]
    d = np.ascontiguousarray(d)
    if train:
        d = resize_and_crop_from_mvsnet(d)
    d = torch.from_numpy(d)
    d.masked_fill_(d == 0, np.nan)
    return d


def resize_and_crop_from_mvsnet(hr_img):
    r"""Resize full-resolution 1600x1200 image to 640x512.

    Source: https://github.com/alibaba/cascade-stereo/blob/master/CasMVSNet/datasets/dtu_yao.py#L67
    """
    # downsample
    h, w = hr_img.shape
    hr_img_ds = cv2.resize(hr_img, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
    # crop
    h, w = hr_img_ds.shape
    target_h, target_w = 512, 640
    start_h, start_w = (h - target_h) // 2, (w - target_w) // 2
    hr_img_crop = hr_img_ds[start_h: start_h + target_h, start_w: start_w + target_w]
    return hr_img_crop


def corrupt_depth(d, focal=2887, sigma_disp=1 / 6, scale=1 / 4):
    r"""Corrupt depth map to simulate Kinect v2.
    Box-downsample the depth to 1/4 of original (RGB) size,
    then add noise with a simplified procedure from https://www.doc.ic.ac.uk/~ahanda/VaFRIC/icra2014.pdf
    then nn-upsample back.

    Parameters
    ----------
    d : torch.Tensor
        Z in mm
    focal : scalar
        Focal length in pixels. Default value is for 1600x1200 depth.
    sigma_disp : scalar
        Std of disparity noise
    scale: scalar
        Scaling factor.
    """
    initial_shape = d.shape
    d = torch.nn.functional.interpolate(d[None, None], scale_factor=scale, mode='area')[0, 0]
    focal = focal * scale

    virtual_baseline = 10  # 10 cm
    baseline_times_focal = virtual_baseline * focal
    d = baseline_times_focal / (baseline_times_focal / d + torch.randn_like(d) * sigma_disp + .5)

    d = torch.nn.functional.interpolate(d[None, None], scale_factor=1 / scale, mode='nearest')[0, 0]
    assert d.shape == initial_shape
    return d


def make_train(depth_root, corrupted_depth_root):
    r""" Generate train/val split for Corrupted DTU.

    Parameters
    ----------
    depth_root : Path
        Directory with `scan{id}` subdirectories, which contain full-resolution depth maps, i.e `depth_map_{cam_id}.pfm`.
    corrupted_depth_root : Path
        Output directory which will contain `scan{id}` subdirs.
    """

    from tqdm import tqdm

    train = True
    seed = 34872566
    torch.manual_seed(seed)
    np.random.seed(seed)

    for scan_dir in tqdm(depth_root.glob('scan*')):
        scan = scan_dir.stem
        if scan.endswith('_train'):
            continue

        for raw_depth in tqdm(scan_dir.glob('depth_map_*.pfm')):
            cam_id = int(raw_depth.stem[-4:])
            d = read_depth(depth_root, scan, cam_id, train)
            d = corrupt_depth(d)
            d.masked_fill_(~torch.isfinite(d), 0)
            d = d.numpy()

            out_path = corrupted_depth_root / (scan + '_train')
            out_path.mkdir(parents=True, exist_ok=True)
            out_path = out_path / raw_depth.name
            imageio.imwrite(out_path, d[::-1])


def make_test(rgb_root, depth_root, corrupted_depth_root):
    r"""Generate test split for Corrupted DTU.

    Parameters
    ----------
    rgb_root : Path
        Directory with `scan{id}` subdirectories, which contain test RGBs, in particular `rect_{cam_id}_max.png`.
    depth_root : Path
        Directory with `scan{id}` subdirectories, which contain full-resolution depth maps, i.e `depth_map_{cam_id}.pfm`.
    corrupted_depth_root : Path
        Output directory which will contain `scan{id}` subdirs.
    """
    from tqdm import tqdm

    seed = 34872566
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = False

    for rgb_dir in tqdm(list(rgb_root.glob('scan*'))):
        scan = rgb_dir.stem
        for rgb in tqdm(list(rgb_dir.glob('rect_*_max.png'))):
            cam_id = int(rgb.stem[5:8]) - 1
            d = read_depth(depth_root, scan, cam_id, train)
            d = corrupt_depth(d)
            d.masked_fill_(~torch.isfinite(d), 0)
            d = d.numpy()

            out_path = corrupted_depth_root / scan
            out_path.mkdir(parents=True, exist_ok=True)
            out_path = out_path / f'depth_map_{cam_id:04}.pfm'
            imageio.imwrite(out_path, d[::-1])
