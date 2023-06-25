#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/21 15:21
# @Author  : huangxiaohan
# @File    : test_data_precision.py
# @Software: PyCharm
'''
Calculate the accuracy of the reconstructed image in the test set
For the reconstructed image block, only the reconstructed region is calculated, and the original real region is not involved in the calculation
'''
import time

import cv2 as cv
import os
import xlsxwriter as xw
from tools_cal_precision import *


def getfilename(path, suffix):
    """ Gets the file with the specified suffix in the specified directory """
    file_list = []
    f_list = os.listdir(path)
    f_list.sort(key=lambda x: str(x[:-4]))  # .tif  Sorted by name   key=lambda x: int(x[:-4]) Sorted by size
    # print f_list
    for file in f_list:
        # os.path.splitext():Separate file name and extension
        if os.path.splitext(file)[1] == suffix:
            file_list.append(os.path.join(path, file))
    return file_list

def lst_evalu1(result_img, ori_img, fac):
    RC = ori_img.shape
    R = RC[0]
    C = RC[1]
    # RMSE
    rmse = 0.0
    num = 0
    for ri in range(R):
        for ci in range(C):
            dif2 = (result_img[ri][ci] - ori_img[ri][ci])
            if dif2>-0.15 and dif2<0.15:
                rmse = rmse + dif2 * dif2
                num = num+1
    rmse = rmse / num
    rmse = math.sqrt(rmse) * fac
    return rmse
def test():
    '''
    Enter two folder paths to calculate the r2 and rmse of the corresponding image
    The DN value range that needs to be scaled to the original image is in K
    rmse = rmse*(max-min)
    '''

    predict_path = 'F:/Night_S4/dataset1/test/predict/'
    label_path = 'F:/Night_S4/dataset1/test/label/'
    mask_path = 'F:/Night_S4/dataset1/train/mask/'
    sta_path = 'F:/test_data/NightS4_rmse2.xlsx'

    workbook = xw.Workbook(sta_path)
    worksheet1 = workbook.add_worksheet('rmse')

    worksheet1.write(0, 0, 'predict_img')
    worksheet1.write(0, 1, 'label_img')

    worksheet1.write(0, 3, 'RMSE')
    worksheet1.write(0, 4, 'R2')

    worksheet1.write(0, 5, 'RMSE_area')
    worksheet1.write(0, 6, 'mean_RMSE')
    worksheet1.write(0, 7, 'mean_RMSE_area')
    worksheet1.write(0, 8, 'mean_R2')

    m1 = 1

    avg_m1_rmse1 = 0
    avg_m1_rmse2 = 0
    avg_m1_r2 = 0

    pred_imgs_names = getfilename(predict_path, '.tif')
    label_imgs_names = getfilename(label_path, '.tif')

    num = len(pred_imgs_names)

    for i in range(num):
        pred_name = pred_imgs_names[i]
        label_name = label_imgs_names[i]
        pred_img = cv.imread(pred_name, cv.IMREAD_UNCHANGED)
        label_img = cv.imread(label_name, cv.IMREAD_UNCHANGED)

        pred_img_name = pred_name.split('/')[-1]
        label_img_name = label_name.split('/')[-1]
        mask_img_name = pred_img_name[0]+'.tif'

        # Find the corresponding mask file
        mask_img = cv.imread(mask_path+mask_img_name, cv.IMREAD_UNCHANGED)
        mask = mask_img/255.

        image = (1 - mask) * pred_img + label_img * mask
        res = (image - label_img) * 100  # Zoom to the DN value range of the original image
        res_nz = res[res.nonzero()]
        num_nz = len(res_nz)
        sqrtt = 0.0
        for j in range(num_nz):
            sqrtt = sqrtt + res_nz[j] * res_nz[j]  # Calculate RMSE
        sqrtt = sqrtt / num_nz
        RMSE_area = np.sqrt(sqrtt)  # Only the rmse of the predicted missing region is calculated

        R2 = performance_metric(label_img, pred_img)  # R2
        RMSE = lst_evalu1(pred_img, label_img, 100)  #Zoom to K

        worksheet1.write(m1, 0, pred_img_name)
        worksheet1.write(m1, 1, label_img_name)
        worksheet1.write(m1, 3, RMSE)  # RMSE
        worksheet1.write(m1, 4, R2)  # R2
        worksheet1.write(m1, 5, RMSE_area)  # SSIM

        m1 = m1 + 1
        avg_m1_rmse1 += RMSE
        avg_m1_rmse2 += RMSE_area
        avg_m1_r2 += R2

        worksheet1.write(2, 6, avg_m1_rmse1/m1)  # SSIM
        worksheet1.write(2, 7, avg_m1_rmse2/m1)  # SSIM
        worksheet1.write(2, 8, avg_m1_r2/m1)  # SSIM

    workbook.close()


if __name__ == '__main__':
    print('In service....')
    start = time.time()
    test()
    end = time.time()
    print('\nEnd of run . Time consuming:{}s'.format(end-start))
