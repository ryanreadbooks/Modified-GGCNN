# This is a modification of GG-CNN

**Note:** This repository is based on the original implementation. **The original implementation** is [here](https://github.com/dougsm/ggcnn).

### Modifications

* Add GraspNet 1Billion dataset. The dataset can be downloaded [here](https://graspnet.net/datasets.html).
* Add deeper GGCNN network.

### Usage

1. You should download the [GraspNet 1Billion](https://graspnet.net/datasets.html) Dataset first. (abbre. *gn1b*)

2. Install graspnetAPI following [here](https://graspnetapi.readthedocs.io/en/latest/install.html#install-api).

   ```bash
   pip install graspnetAPI
   ```   

3. Change the `graspnet_root` in script `convert_gn1b_2_cornell.py`, then run it to convert the dataset format into Cornell Dataset format.

4. Run the training script `train_ggcnn.py` to train the model using *gn1b* Dataset.

   #### Example

   ```bash
   python train_ggcnn.py --network ggcnn2 --dataset graspnet1b --dataset-path path-to-gn1b --camera realsense --epochs 20 --batch-size 16 --num-workers 20 --description ggcnn2_gn1b
   ```

### Reference

[GG-CNN](https://github.com/dougsm/ggcnn)

[GraspNet Dateset](https://graspnet.net/datasets.html)



