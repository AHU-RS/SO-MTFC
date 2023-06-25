

import numpy as np


'''
Batch reconstruction of different date images
'''
from model import *
import cv2 as cv
import os
import shutil

def get_data(path, name):
    t1_names = os.listdir(path + name)
    t1_names.sort(key=lambda x: int(x[:-4]))
    T1 = []


    num = len(t1_names)
    for i in range(num):
        t1_name = t1_names[i]
        t1_img = cv.imread(path + name + t1_name, cv.IMREAD_UNCHANGED)
        t1_img = t1_img.reshape([24, 24, 1])
        T1.append(t1_img)
    T1 = np.array(T1)
    return T1

# Returns a list of time folders
def get_dir(path):
    file_list = []
    dir_list = os.listdir(path)
    for file in dir_list:
        file_list.append(os.path.join(path, file))
    return file_list


# Input path
path = 'F:/Day_S3/Day_S3_crop_r/'
dir_list = get_dir(path)
num = len(dir_list)
# Trained model file
checkpoint_save_path = "F:/Day_S3/dataset5/checkpoint/DayS3_MSE_300.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model = MTFC_model()
    model.load_weights(checkpoint_save_path)


for n in range(1, len(dir_list)-1):
    print('{}being processed...'.format(dir_list[n]))
    t1_path = dir_list[n - 1]
    t2_path = dir_list[n]
    t3_path = dir_list[n + 1]

    # Use the predicted value at time T1 as the T1 output
    # Note that all_pred/ on the first day is set as the interpolation result, namely all_chazhi/
    t1_imgs = get_data(t1_path, '/all_queshi/')
    t2_imgs = get_data(t2_path, '/all_pred/')
    t3_imgs = get_data(t3_path, '/all_queshi/')

    concat_1 = tf.concat([t1_imgs, t3_imgs], 3)
    add_1 = tf.concat([concat_1, t2_imgs], 3)
    add_2 = concat_1 + t2_imgs
    test = tf.concat([add_1, add_2], 3)
    test_img = np.array(test)


    result = model.predict(test_img)  # (1667,24,24,1)

    out_path = t2_path + '/all_pred/'
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    f_list = os.listdir(t2_path + '/all_queshi/')
    f_list.sort(key=lambda x: int(x[:-4]))

    # Save the predicted image
    for i in range(len(result)):
        # Use a mask! The T2 image was masked
        name = out_path + f_list[i]
        pred_img = result[i,:,:,0]
        t2_img = t2_imgs[i,:,:,0]
        t2_img_mask = np.where(t2_img > 0, 1, 0)  # To make a mask, the missing part is 0 and the missing part is 1
        img = (1 - t2_img_mask) * pred_img + t2_img * t2_img_mask
        cv.imwrite(name, img)
