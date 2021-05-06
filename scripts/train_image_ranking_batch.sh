#!/usr/bin/env bash
exp_id="img_ranking_batchhard_efficientnet"

source "scripts/master_env.sh"

python main.py \
    --gpu_id $GPUID \
    -w $N_WORKERS \
    --dataset_cfg "./configs/dataset_cfgs/tiny_imagenet_reid_batch.yaml" \
    --train_cfg   "./configs/train_cfgs/${exp_id}.yaml" \
    --logdir      "logs/${exp_id}" \
    --log_fname   "logs/${exp_id}/${exp_id}_train.log" \
    --train_mode  "from_scratch" \
    --is_training true