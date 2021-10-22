import os
import subprocess

from graspnetAPI import graspnet

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

graspnet_root = '/share/home/ryan/datasets/graspnet-1billion'


for scene_id in range(190):
    for camera in ('realsense', 'kinect'):
        scene_str = f'scene_{str(scene_id).zfill(4)}'
        cornell_dest_dir = os.path.join(graspnet_root, 'scenes', scene_str, camera, 'rect_cornell')   # cornell目标地址
        original_dest_dir = os.path.join(graspnet_root, 'scenes', scene_str, camera, 'rect')   # 自带的rect label的目标地址
        cornell_source_dir = os.path.join(graspnet_root, 'rect_labels_cornell', scene_str, camera)  # cornell的源地址
        original_source_dir = os.path.join(graspnet_root, 'rect_labels', scene_str, camera)    # 自带的rect label的源地址
        ensure_dir(original_dest_dir)   # 创建文件夹
        ensure_dir(cornell_dest_dir)    # 创建文件夹

        # os.system(f'rm -rf {cornell_dest_dir}')
        # os.system(f'rm -rf {original_dest_dir}')

        print(cornell_source_dir, ' ==> ', cornell_dest_dir)
        print(original_source_dir, ' ==> ', original_dest_dir)
        # command = f'cp {original_source_dir}/*.npy {original_dest_dir}/'
        # os.system(command)
        command1 = f'cp {cornell_source_dir}/*.txt {cornell_dest_dir}/'
        os.system(command1)
        print('===')
