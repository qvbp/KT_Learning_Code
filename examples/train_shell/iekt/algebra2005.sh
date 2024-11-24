#!/bin/bash

# Common configurations
dataset_name="algebra2005"
emb_type="qid"
emb_sizes=(256 256 256 256 256)
dropouts=(0.5 0.3 0.1 0.1 0.3)
n_layers=(1 1 1 1 1)
cog_levels=(20 20 10 20 10)
gamma=(0.8 0.8 0.4 0.6 0.8)
lambs=(30 50 30 30 50)
seeds=(3407 3407 3407 3407 42)
acq_levels=(20 10 30 10 30)
learning_rate=0.001

# Iterate over each fold
for fold in {0..4}; do
    CUDA_VISIBLE_DEVICES=$fold python ./wandb_iekt_train.py \
        --dataset_name=$dataset_name \
        --fold=$fold \
        --emb_type=$emb_type \
        --emb_size=${emb_sizes[$fold]} \
        --dropout=${dropouts[$fold]} \
        --n_layer=${n_layers[$fold]} \
        --cog_levels=${cog_levels[$fold]} \
        --gamma=${gamma[$flod]} \
        --lamb=${lambs[$fold]} \
        --seed=${seeds[$fold]} \
        --acq_levels=${acq_levels[$fold]} \
        --learning_rate=$learning_rate \
        > ./train_log/iekt/iekt_algebra2005_${fold}_train.txt &
done
