#！/bin/sh

python train_ggcnn.py \
  --network ggcnn \
  --dataset graspnet1b \
  --dataset-path /share/home/ryan/datasets/graspnet-1billion \
  --use-rgb 1 \
  --output-size 480 \
  --camera realsense \
  --scale 1 \
  --num-workers 20 \
  --device 1 \
  --batch-size 32 \
  --epochs 50 \
  --batches-per-epoch 800 \
  --val-batches 300 \
  --description ggcnn \
  --outdir new_output/models \
  --logdir new_tensorboard \

