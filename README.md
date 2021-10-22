# This is a modification of GG-CNN

**Note:** This repository is based on the original implementation. **The original implementation** is [here](https://github.com/dougsm/ggcnnhttps://github.com/dougsm/ggcnn).

### Modifications

* Add GraspNet 1Billion dataset. The dataset can be downloaded [here][https://graspnet.net/datasets.html].
* Add deeper GGCNN network.

### Usage

1. You should download the [GraspNet 1Billion](https://graspnet.net/datasets.html) Dataset first. (Abbre. gn1b)

2. Install graspnetAPI.

   ```bash
   pip install graspnetAPI
   ```

3. Add the following code into `graspnetAPI/grasp.py/RectGraspGroup`, make it a member function of class `RectGraspGroup` in graspnetAPI

   ```python
   def get_all_rect_corners(self):
           """
          	get the corners of all grasp rectangle
          	return: np.ndarray, shape=(n, 8) => [x1,y1,x2,y2,x3,y3,x4,y4]
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
   ```

   

4. Run the script `convert_gn1b_2_cornell.py` to convert the dataset format into Cornell Dataset format.

5. Run the training script `train_ggcnn.py` to train the model using GraspNet 1Billion Dataset.

   #### Example

   ```bash
   python train_ggcnn.py --network ggcnn2 --dataset graspnet1b --dataset-path path-to-gn1b dataset --camera realsense --epochs 20 --batch-size 16 --num-workers 20 --description ggcnn2_gn1b
   ```

### Reference

[GG-CNN](https://github.com/dougsm/ggcnn)

[GraspNet Dateset](https://graspnet.net/datasets.html)



