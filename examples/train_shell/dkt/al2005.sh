#!/bin/bash

# Common configurations
dataset_name="algebra2005"
emb_type="qid"
emb_sizes=(256 256 256 256 256)
dropouts=(0.5 0.5 0.5 0.1 0.3)
seeds=(42 42 3407 42 3407)
learning_rate=0.001

# Iterate over each fold
for fold in {0..4}; do
    CUDA_VISIBLE_DEVICES=$fold python ./wandb_dkt_train.py \
        --dataset_name=$dataset_name \
        --fold=$fold \
        --emb_type=$emb_type \
        --emb_size=${emb_sizes[$fold]} \
        --dropout=${dropouts[$fold]} \
        --seed=${seeds[$fold]} \
        --learning_rate=$learning_rate \
        > ./train_log/dkt/dkt_change_algebra2005_${fold}_train.txt &
done
