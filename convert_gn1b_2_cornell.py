"""
将GraspNet-1Billion的抓取矩形框标注转换成Cornell数据集中的标注格式
"""

import os

from numpy.lib.polynomial import roots

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


import graspnetAPI
import numpy as np

####################################################################
graspnet_root = '/share/home/ryan/datasets/graspnet-1billion'  # ROOT PATH FOR GRASPNET
####################################################################

from graspnetAPI import GraspNet
from graspnetAPI.grasp import RectGraspGroup

# initialize a GraspNet instance

def run(args):
    scene_start, scene_end = args
    for camera in ('realsense', 'kinect'):
        g = GraspNet(graspnet_root, camera=camera, split='all')
        for scene_id in range(scene_start, scene_end):
            for ann_id in range(256):
                # load rect grasp from graspnet-1b
                rect_grasps: RectGraspGroup = g.loadGrasp(sceneId=scene_id, camera=camera, annId=ann_id, format='rect')
                # sort them
                rect_grasps = rect_grasps.sort_by_score()
                n = (int) (len(rect_grasps) * 0.05) # only use top-5% grasps
                rect_grasps = rect_grasps[:n]
                corners = rect_grasps.get_all_rect_corners()    # (n, 8)
                corners = corners.reshape(-1, 2) # (n * 4, 2)
                # write it into file,
                dir = os.path.join(graspnet_root, 'scenes', f'scene_{str(scene_id).zfill(4)}', camera, 'rect_cornell')
                ensure_dir(dir)
                saved_path = os.path.join(dir, f'{str(ann_id).zfill(4)}.txt')
                print(saved_path)
                np.savetxt(saved_path, corners, fmt='%.2f')


if __name__ == '__main__':
    from multiprocessing import Pool

    args = [(i * 5,(i + 1) * 5) for i in range(38)]

    with Pool(processes=38) as pool:
        pool.map(run, args)
        pool.close()
        pool.join()

    print('Done')
    