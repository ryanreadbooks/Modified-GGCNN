"""
将GraspNet-1Billion的抓取矩形框标注转换成Cornell数据集中的标注格式
"""

from types import MethodType
print(MethodType)
import os
import copy
import numpy as np

from graspnetAPI import GraspNet
from graspnetAPI.grasp import RectGraspGroup
from pandas.api import types

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_all_rect_corners(self):
        """
        所有抓取矩形框中的四个角点的坐标
        格式 np.ndarray, shape=(n, 8) => [x1,y1,x2,y2,x3,y3,x4,y4]

        """
        shuffled_rect_grasp_group_array = copy.deepcopy(self.rect_grasp_group_array)
        corners = np.zeros((0, 8))
        for rect_grasp_array in shuffled_rect_grasp_group_array:
            center_x, center_y, open_x, open_y, height, score, object_id = rect_grasp_array
            center = np.array([center_x, center_y])
            left = np.array([open_x, open_y])
            axis = left - center
            normal = np.array([-axis[1], axis[0]])
            normal = normal / np.linalg.norm(normal) * height / 2
            p1 = center + normal + axis
            p2 = center + normal - axis
            p3 = center - normal - axis
            p4 = center - normal + axis
            corner = np.hstack([p1, p2, p3, p4])
            corners = np.vstack([corners, corner])
        return corners

RectGraspGroup.get_all_rect_corners = get_all_rect_corners
####################################################################
graspnet_root = '/share/home/ryan/datasets/graspnet-1billion'  # ROOT PATH FOR GRASPNET
####################################################################



# initialize a GraspNet instance

def run(args):
    scene_start, scene_end = args
    for camera in ('realsense', 'kinect'):
        g = GraspNet(graspnet_root, camera=camera, split='train')
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

    n = 38
    args = [(i * 5,(i + 1) * 5) for i in range(n)]

    with Pool(processes=n) as pool:
        pool.map(run, args)
        pool.close()
        pool.join()

    print('Done')
    