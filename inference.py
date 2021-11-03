"""
written by ryanreadbooks
date: 2021/10/17
"""
from os import path
import sys

import cv2
import imageio
import numpy as np
import scipy.ndimage as ndimage
from skimage.feature import peak_local_max
import torch

from utils.timeit import TimeIt
from utils.dataset_processing.evaluation import plot_output
from utils.dataset_processing.grasp2 import Grasp
from utils.data.grasp_data import GraspDatasetBase
from models import *

# MODEL_FILE = 'output/models/211017_1513_/ggcnn_wo_rgb_epoch_49_iou_0.77'   # ggcnn
MODEL_FILE = 'output/models/211017_1535_/epoch_04_iou_0.94'   # ggcnn2
here = path.dirname(path.abspath(__file__))
sys.path.append(here)
print(path.join(path.dirname(__file__), MODEL_FILE))
model = torch.load(path.join(path.dirname(__file__), MODEL_FILE))
device = torch.device("cuda:0")


def process_depth_image(depth, crop_size, out_size=300, return_mask=False, crop_y_offset=0):
    imh, imw = depth.shape

    with TimeIt('1'):
        # Crop.
        depth_crop = depth[(imh - crop_size) // 2 - crop_y_offset: (imh - crop_size) // 2 + crop_size - crop_y_offset,
                     (imw - crop_size) // 2:(imw - crop_size) // 2 + crop_size]
    # depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)

    # Inpaint
    # OpenCV inpainting does weird things at the border.
    with TimeIt('2'):
        depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)

    with TimeIt('3'):
        depth_crop[depth_nan_mask == 1] = 0

    with TimeIt('4'):
        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        depth_scale = np.abs(depth_crop).max()
        depth_crop = depth_crop.astype(np.float32) / depth_scale  # Has to be float32, 64 not supported.

        with TimeIt('Inpainting'):
            depth_crop = cv2.inpaint(depth_crop, depth_nan_mask, 1, cv2.INPAINT_NS)

        # Back to original size and value range.
        depth_crop = depth_crop[1:-1, 1:-1]
        depth_crop = depth_crop * depth_scale

    with TimeIt('5'):
        # Resize
        depth_crop = cv2.resize(depth_crop, (out_size, out_size), cv2.INTER_AREA)

    if return_mask:
        with TimeIt('6'):
            depth_nan_mask = depth_nan_mask[1:-1, 1:-1]
            depth_nan_mask = cv2.resize(depth_nan_mask, (out_size, out_size), cv2.INTER_NEAREST)
        return depth_crop, depth_nan_mask
    else:
        return depth_crop


def process_rgb(rgb, crop_size, out_size, crop_y_offset=0):
    """
    rgb: np.ndarray (h,w,3)
    """
    imh, imw, c = rgb.shape

    with TimeIt('1'):
        # Crop.
        rgb_crop = rgb[(imh - crop_size) // 2 - crop_y_offset: (imh - crop_size) // 2 + crop_size - crop_y_offset,
                   (imw - crop_size) // 2: (imw - crop_size) // 2 + crop_size]
        rgb_crop = cv2.resize(rgb_crop, (out_size, out_size), cv2.INTER_AREA)

    # normalise
    rgb_crop = np.asarray(rgb_crop, dtype=np.float32) / 255.0
    rgb_crop -= rgb_crop.mean()
    rgb_crop = rgb_crop.transpose((2, 0, 1))

    return rgb_crop  # (h,w,3)


def predict(depth, rgb=None, process_depth=True, crop_size=300, out_size=300, depth_nan_mask=None, crop_y_offset=0, filters=(2.0, 1.0, 1.0)):
    if process_depth:
        depth, depth_nan_mask = process_depth_image(depth, crop_size, out_size=out_size, return_mask=True, crop_y_offset=crop_y_offset)

    # Inference
    depth = np.clip((depth - depth.mean()), -1, 1)
    depthT = torch.from_numpy(depth.reshape(1, 1, out_size, out_size).astype(np.float32)).to(device)
    network_input = depthT
    if rgb is not None:
        # 包括了rgb信息
        rgb = process_rgb(rgb, crop_size, out_size=out_size, crop_y_offset=crop_y_offset)
        rgb = torch.from_numpy(rgb)[None].to(device)   # (1,3,h,w)
        network_input = torch.cat([depthT, rgb], dim=1)

    with torch.no_grad():
        pred_out = model(network_input)

    points_out = pred_out[0].cpu().numpy().squeeze()
    points_out[depth_nan_mask] = 0

    # Calculate the angle map.
    cos_out = pred_out[1].cpu().numpy().squeeze()
    sin_out = pred_out[2].cpu().numpy().squeeze()
    ang_out = np.arctan2(sin_out, cos_out) / 2.0

    width_out = pred_out[3].cpu().numpy().squeeze() * 150.0  # Scaled 0-150:0-1

    # Filter the outputs.
    if filters[0]:
        points_out = ndimage.filters.gaussian_filter(points_out, filters[0])  # 3.0
    if filters[1]:
        ang_out = ndimage.filters.gaussian_filter(ang_out, filters[1])
    if filters[2]:
        width_out = ndimage.filters.gaussian_filter(width_out, filters[2])

    points_out = np.clip(points_out, 0.0, 1.0 - 1e-3)

    # SM
    # temp = 0.15
    # ep = np.exp(points_out / temp)
    # points_out = ep / ep.sum()

    # points_out = (points_out - points_out.min())/(points_out.max() - points_out.min())

    return points_out, ang_out, width_out, depth.squeeze()


def detect_grasps(point_img, ang_img, width_img=None, num_grasps=1, ang_threshold=5, thresh_abs=0.5, min_distance=20):
    local_max = peak_local_max(point_img, min_distance=min_distance, threshold_abs=thresh_abs, num_peaks=num_grasps)

    grasps = []

    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)

        grasp_angle = ang_img[grasp_point]
        if ang_threshold > 0:
            if grasp_angle > 0:
                grasp_angle = ang_img[grasp_point[0] - ang_threshold:grasp_point[0] + ang_threshold + 1,
                              grasp_point[1] - ang_threshold:grasp_point[1] + ang_threshold + 1].max()
            else:
                grasp_angle = ang_img[grasp_point[0] - ang_threshold:grasp_point[0] + ang_threshold + 1,
                              grasp_point[1] - ang_threshold:grasp_point[1] + ang_threshold + 1].min()

        g = Grasp(grasp_point, grasp_angle, value=point_img[grasp_point])
        if width_img is not None:
            g.length = width_img[grasp_point]
            g.width = g.length / 2

        grasps.append(g)

    return grasps


if __name__ == '__main__':
    from utils.dataset_processing.image import Image, DepthImage

    # depth_img_path = '/home/ryan/Datasets/my_images4grasp/realworld_data/0020_depth.png'
    # rgb_img_path = '/home/ryan/Datasets/my_images4grasp/realworld_data/0020_color.jpg'
    scene_id = 518
    depth_img_path = f'/home/ryan/Datasets/my_images4grasp/bop_data/ycbv/train_pbr/000000/depth/{str(scene_id).zfill(6)}.png'
    rgb_img_path = f'/home/ryan/Datasets/my_images4grasp/bop_data/ycbv/train_pbr/000000/rgb/{str(scene_id).zfill(6)}.jpg'

    rgb_im = imageio.imread(rgb_img_path)
    depth = imageio.imread(depth_img_path)

    crop_size = 300
    imh, imw = depth.shape
    rgb_im_Image = Image(rgb_im)
    rgb_im_arr = np.asarray(rgb_im_Image, dtype=np.uint8)
    rgb_im_crop = rgb_im_arr[(imh - crop_size) // 2: (imh - crop_size) // 2 + crop_size,
                  (imw - crop_size) // 2:(imw - crop_size) // 2 + crop_size]

    depth_arr = np.asarray(depth).astype(np.float64)
    depth_arr *= 0.0001  # 单位化成m

    points_out, ang_out, width_out, depth_squeeze = predict(depth_arr, rgb_im_arr)  # 得到三个分支的输出

    # grasps = detect_grasps(points_out, ang_out, width_out, )

    plot_output(rgb_im_crop, depth_squeeze, points_out, ang_out, no_grasps=10, grasp_width_img=width_out)

    print('done')
