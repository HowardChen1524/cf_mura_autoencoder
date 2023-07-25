#!/bin/bash


data_dir='/home/mura/Min/localization/cf_mura_rmeg/0.01_20/fin_res'
csv_dir='/home/mura/Min/localization/AutoEncoder_mura_copy/detect_position/cf_mura_b2/cf_mura_b2.csv'
save_dir='/home/mura/Min/localization/cf_mura_rmeg/0.01_20/fin_res_gt'
isResize=1

mkdir -p $save_dir

python plot_gt_on_result.py \
-dd=$data_dir \
-cp=$csv_dir \
-sd=$save_dir \
-rs=$isResize