#!/bin/bash
# use_spectral_norm_D whether to add spectral norm to D, it helps improve results

input="/home/mura/mura_data/cf_mura/normal"
savedir="/home/mura/AutoEncoder/NYCU_pytorch_AE/model"
testname="conv_1024_3_layers_32_16_8_k5k3k5_CyclicLR"
epochs=200
batchs=64
lr=0.001
num_workers=6
encodesize=1024
decodesize=256
devices="0"

python train_conv.py \
 --input=$input  --savedir=$savedir --testname=$testname \
 --epochs=$epochs --batchs=$batchs --num_workers=$num_workers \
 --encodesize=$encodesize --decodesize=$decodesize \
 --devices=$devices