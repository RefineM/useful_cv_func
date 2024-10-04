#!/bin/bash

# train and render
for i in {1..50}
do 
    # free gpu 
    nvidia-smi -i 2 | grep 'python'| cut -c 21-27 | xargs kill -9

    echo "##################### training: $i #######################"
    interval=700
    n_steps=$(($i * $interval))
    flag="$i"
    csv_dir="ngp_output/20240511_6240_74_log/"
    exp_dir="ngp_output/20240511_6240_74_log/$i"
    save_snapshot="$exp_dir/model.msgpac"
    
    echo "model save to: $save_snapshot"

    CUDA_VISIBLE_DEVICES=2 python instant-ngp/scripts/run.py \
    --scene ngp_data/20240511_6240_74 \
    --n_steps $n_steps \
    --save_snapshot $save_snapshot \
    --train_transforms ngp_data/20240511_6240_74/train.json \
    --test_transforms ngp_data/20240511_6240_74/eval.json \
    --exp_dir $exp_dir \
    --csv_dir $csv_dir 
    
    sleep 3

done



