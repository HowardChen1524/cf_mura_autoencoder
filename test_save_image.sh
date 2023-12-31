#!/bin/bash
# use_spectral_norm_D whether to add spectral norm to D, it helps improve results

# input="/home/mura/mura_data/cf_mura/normal_testing"
input="/home/mura/mura_data/cf_mura/mura"

modelpath="/home/mura/Min/localization/model/conv_1024_3_layers_32_16_8_k3s2p1_CyclicLR/200.pth"
imagepath="../cf_mura_rmeg/"

batchs=1
num_workers=6
encodesize=1024
devices="0"

for th_percent in $(seq 0.01 0.01 0.01)
do
    for min_area in $(seq 1 1 1)
    do
        python test_save_image.py \
        --input=$input  --modelpath=$modelpath --imagepath=$imagepath \
        --batchs=$batchs --num_workers=$num_workers \
        --encodesize=$encodesize \
        --devices=$devices \
        --th_percent=$th_percent --min_area=$min_area
    done
done

