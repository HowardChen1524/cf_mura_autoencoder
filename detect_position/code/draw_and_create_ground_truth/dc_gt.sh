#!/bin/bash
# dataset_version='typec+b1'
# dataset_version='typed'
dataset_version='typec+b2'
data_dir='/home/mura/mura_data/'
save_dir='/home/mura/Min/localization/Mura_ShiftNet-main/detect_position/'

python dc_gt.py \
-dv=$dataset_version \
-dd=$data_dir \
-sd=$save_dir