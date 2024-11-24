#!/bin/bash

# Common configurations
dataset_name="bridge2algebra2006"
emb_type="qid"
loss3=(0.5 0.5 0.5 0.5 0.5)
d_ff=(256 64 64 256 64)
nheads=(4 4 4 4 4)
dropouts=(0.1 0.1 0.1 0.1 0.1)
loss2=(0.5 0.5 0.5 0.5 0.5)
final_fc_dim2=(64 64 256 256 256)
loss1=(0.5 0.5 0.5 0.5 0.5)
d_model=(256 256 256 256 256)
num_attn_heads=(4 8 8 8 4)
num_layers=(2 2 2 2 2)
seeds=(42 3407 3407 3407 42)
final_fc_dim=(256 64 64 256 256)
n_blocks=(4 4 4 4 4)
start=(50 50 50 50 50)
learning_rate=(0.00001 0.0001 0.0001 0.0001 0.00001)

# Iterate over each fold
for fold in {0..4}; do
    CUDA_VISIBLE_DEVICES=$fold python ./wandb_simplekt_train.py \
        --dataset_name=$dataset_name \
        --fold=$fold \
        --emb_type=$emb_type \
        --loss3=${loss3[$fold]} \
        --d_ff=${d_ff[$fold]} \
        --nheads=${nheads[$fold]} \
        --dropout=${dropouts[$fold]} \
        --loss2=${loss2[$fold]} \
        --final_fc_dim2=${final_fc_dim2[$fold]} \
        --loss1=${loss1[$fold]} \
        --d_model=${d_model[$fold]} \
        --num_attn_heads=${num_attn_heads[$fold]} \
        --num_layers=${num_layers[$fold]} \
        --seed=${seeds[$fold]} \
        --final_fc_dim=${final_fc_dim[$fold]} \
        --n_blocks=${n_blocks[$fold]} \
        --start=${start[$fold]} \
        --learning_rate=${learning_rate[$fold]} \
        > ./train_log/simplekt/simplekt_change_assist2009_${fold}_train.txt &
done
