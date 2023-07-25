import torch
import torch.nn as nn
from model import AutoEncoderConv
from dataloader import mura_dataloader
from option import parse_option
from statistics import mean, stdev
import os
import numpy as np
import cv2
import time

opts = parse_option()

# ===== function =====
# Show images
def save_images(img_index, img_name, images, save_path):
    for index, image in enumerate(images):
        array = image.reshape(1024, 1024) * 255
        cv2.imwrite(os.path.join(save_path, img_name), array)

def save_image_howard(img_index, img_name, image, save_dir, dtype):
    save_path = os.path.join(save_dir, dtype)
    os.makedirs(save_path, exist_ok=True)
    array = image.reshape(1024, 1024) * 255
    cv2.imwrite(os.path.join(save_path, img_name), array)

# dirt 可能會有狀況
def get_frame_region(imga, erode=5, dilate=21, debug=False):
    imga = np.transpose(imga, (1, 2, 0))[..., 0]
    mask = imga*255>=180
    mask = mask | (imga==0)
    mask = cv2.erode(np.uint8(mask), np.ones((erode, erode)))
    mask = (mask>0)|(imga*255>=250)
    mask = cv2.dilate(np.uint8(mask), np.ones((dilate, dilate)))
    n_components, components, stats, _ = cv2.connectedComponentsWithStats(mask)
    for n in range(1, n_components):
        # if stats[n][-1] < 10000:
        if stats[n][-1] < 3151:
            mask[components==n] = 0
    return mask==0 

def remove_small_areas_opencv(image):
    image = image.astype(np.uint8)
    
    # 使用 connectedComponents 函數
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    
    # 指定面積閾值
    min_area_threshold = opts.min_area
    
    # 遍歷所有區域
    for i in range(1, num_labels):
        # 如果區域面積小於閾值，就將對應的像素值設置為黑色
        if stats[i, cv2.CC_STAT_AREA] < min_area_threshold:
            labels[labels == i] = 0
    
    # 將標籤為 0 的像素設置為白色，其它像素設置為黑色
    result = labels.astype('uint8')
    # print(np.unique(labels))
    result[result == 0] = 0
    result[result != 0] = 1
    return result

# ===================

# create save path
save_dir = os.path.join(opts.imagepath, f'{str(opts.th_percent)}_{opts.min_area}')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Result save path: {save_dir}")

# set devoce
device = torch.device(f"cuda:{opts.devices}" if (torch.cuda.is_available()) else "cpu")
print(f'Using {device} for inference')

# Load the saved model
model = AutoEncoderConv().to(device)
model.load_state_dict(torch.load(opts.modelpath, map_location=device))
model.eval()

# Create a DataLoader for the test dataset
test_dataloader = mura_dataloader(opts)
print("Number of testing data: ", len(test_dataloader))

# loss function
mse_loss = nn.MSELoss(reduction='none').to(device)

all_MSE = []
excpet_list = []
# Iterate over the test dataset using the DataLoader
with torch.no_grad():
    number_of_image = 1
    for i, data in enumerate(test_dataloader):
        img = data[0]
        img_name = data[1][0]
        print(f'{time.ctime()} -> Image {number_of_image}: {img_name}')
        
        input = img.to(device) 
        prediction = model(input)

        prediction = prediction.detach().cpu().numpy()
        input = input.detach().cpu().numpy()
        
        # rm edge
        try:
            mask = get_frame_region(input[0])

            input = np.transpose(input[0], (1, 2, 0))[..., 0]
            prediction = np.transpose(prediction[0], (1, 2, 0))[..., 0]
            input[mask==0] = 0
            prediction[mask==0] = 0
            
            # save_image_howard(i, img_name, mask, save_dir, 'mask')
            # save_image_howard(i, img_name, input, save_dir, 'ori')
            # save_image_howard(i, img_name, prediction, save_dir, 'pred')

            # diff = mse_loss(prediction, input)[0]
            diff = np.square(prediction - input)
            num_pixels = diff[mask!=0].flatten().shape[0] # only panel
            num_top_pixels = int(num_pixels * opts.th_percent)
            filter = np.partition(diff[mask!=0].flatten(), -num_top_pixels)[-num_top_pixels]
            print(f"Panel pixel: {num_pixels}")
            print(f"Threshold: {filter}")
            diff = (diff >= filter)
            # save_image_howard(i, img_name, diff, save_dir, 'ori_res')
            # remove min_area

            res = remove_small_areas_opencv(diff)
            save_image_howard(i, img_name, res, save_dir, 'fin_res')
            
            mse = diff.mean().item()
            print("mse:", mse)
            all_MSE.append(mse)
        except:
            excpet_list.append(img_name)
            print('raise except')
            continue
        # 取百分比 (去邊的前處理，需算mask黑色以外的面積的前幾％)
        # num_pixels = diff.numel()
        # num_top_pixels = int(num_pixels * top_k)
        # filter, _ = diff.view(-1).kthvalue(num_pixels - num_top_pixels)
        # print(f"Theshold: {filter}")

        # 二值化
        # diff[diff>=filter] = 1
        # diff[diff<filter] = 0
        
        # save_images(i, img_name[0], diff, save_dir)
        # save_images(i, img_name[0], prediction, save_dir)

        del prediction, input
        # torch.cuda.empty_cache()
        number_of_image+=1

    print(f'Mean MSE Loss: {mean(all_MSE)}')
    print(f'Standard deviation MSE Loss: {stdev(all_MSE)}')

    MSE_loss_path = os.path.join(save_dir, "test_result.txt")
    MSE_loss_file = open(MSE_loss_path, 'w')

    MSE_loss_file.write(f'Mean MSE Loss: {mean(all_MSE)}\n')
    MSE_loss_file.write(f'Standard deviation MSE Loss: {stdev(all_MSE)}\n')
    MSE_loss_file.write(f'Except imgs: {len(excpet_list)}\n')
    for exc_img in excpet_list:
        MSE_loss_file.write(f'{exc_img}\n')
    MSE_loss_file.close()