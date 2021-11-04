#！/bin/sh

python train_ggcnn.py \
  --network ggcnn4 \
  --dataset cornell \
  --dataset-path /home/ryan/1DataRepository/Datasets/cornell_grasp_dataset \
  --use-rgb 1 \
  --output-size 300 \
  --camera realsense \
  --scale 1 \
  --ggcnn3backend dla34up \
  --num-workers 12 \
  --device 0 \
  --batch-size 8 \
  --epochs 50 \
  --batches-per-epoch 1000 \
  --val-batches 250 \
  --description ggcnn4_unet_w_rgb \
  --outdir new_output/models \
  --logdir new_tensorboard \

