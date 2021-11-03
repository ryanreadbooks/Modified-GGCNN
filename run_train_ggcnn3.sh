#！/bin/sh

python train_ggcnn.py \
  --network ggcnn3 \
  --dataset cornell \
  --dataset-path /home/ryan/1DataRepository/Datasets/cornell_grasp_dataset \
  --use-rgb 0 \
  --output-size 384 \
  --camera realsense \
  --scale 1 \
  --ggcnn3backend dla34up \
  --num-workers 12 \
  --device 0 \
  --batch-size 8 \
  --epochs 50 \
  --batches-per-epoch 1000 \
  --val-batches 250 \
  --description ggcnn3_dla34up_wo_rgb \
  --outdir new_output/models \
  --logdir new_tensorboard \

