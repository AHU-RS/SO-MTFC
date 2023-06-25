import shutil

from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as pltim
import cv2 as cv
import os
'''
Note that the ordering of read files corresponds to the problem; get_file_name get_data
Note the use of mask, reconstruct the area + the original T2 image
'''

from model import *

def get_filename(path,suffix):
    """Gets the file with the specified suffix in the specified directory"""
    file_list = []
    f_list = os.listdir(path)
    f_list.sort(key=lambda x:str(x[:-4])) # .tif    Notice that we use int to sort numbers here; Others are sorted by str
    for file in f_list:
        if os.path.splitext(file)[1] == suffix:
            file_list.append(os.path.join(path,file))
    return file_list
def get_data(t1_path,t2_path,t3_path,label_path):
    """Get sample dataset: T1 T2 T3 label"""
    img_size = 24

    t1_names = get_filename(t1_path, '.tif')
    t2_names = get_filename(t2_path, '.tif')
    t3_names = get_filename(t3_path, '.tif')
    label_names = get_filename(label_path, '.tif')

    T1 = []
    T2 = []
    T3 = []
    Label = []


    num = len(label_names)
    for i in range(num):
        t1_name = t1_names[i]
        t2_name = t2_names[i]
        t3_name = t3_names[i]
        label_name = label_names[i]

        # 读入数据维度(24,24)
        t1_img = cv.imread(t1_name, cv.IMREAD_UNCHANGED)
        t2_img = cv.imread(t2_name, cv.IMREAD_UNCHANGED)
        t3_img = cv.imread(t3_name, cv.IMREAD_UNCHANGED)
        label_img = cv.imread(label_name, cv.IMREAD_UNCHANGED)
        t1_img = t1_img.reshape([img_size, img_size,1])
        t2_img = t2_img.reshape([img_size, img_size, 1])
        t3_img = t3_img.reshape([img_size, img_size, 1])
        label_img = label_img.reshape([img_size, img_size, 1])

        T1.append(t1_img)
        T2.append(t2_img)
        T3.append(t3_img)
        Label.append(label_img)

    T1 = np.array(T1)
    T2 = np.array(T2)
    T3 = np.array(T3)
    Label = np.array(Label)

    # (num,24,24,1)
    return T1,T2,T3,Label



t1_path = 'F:/Day_S1/dataset5/tt/t1/'
t2_path = 'F:/Day_S1/dataset5/tt/t2/'
t3_path = 'F:/Day_S1/dataset5/tt/t3/'
label_path = 'F:/Day_S1/dataset5/tt/t1/'

out_path1 = 'F:/Day_S1/dataset5/tt/out/'
out_path2 = 'F:/Day_S1/dataset5/tt/out/'

if os.path.exists(out_path1):
    shutil.rmtree(out_path1)
os.makedirs(out_path1)
if os.path.exists(out_path2):
    shutil.rmtree(out_path2)
os.makedirs(out_path2)

checkpoint_save_path = "F:/Day_S1/dataset5/checkpoint/DayS1.ckpt"
t1, t2, t3, label = get_data(t1_path, t2_path, t3_path, label_path)
# input layer
concat_1 = tf.concat([t1, t3], 3)  # [1,24,24,2]
add_1 = tf.concat([concat_1, t2], 3)  # [1,24,24,2] + [1,24,24,1] =  [1,24,24,3]
add_2 = concat_1 + t2  # [1,24,24,2]
test = tf.concat([add_1, add_2], 3)  # [1,24,24,5] # Input the model size
test_img = np.array(test)

model = MTFC_model()
model.load_weights(checkpoint_save_path)

result = model.predict(test_img)  # (,24,24,1)

f_list = os.listdir(t2_path)
f_list.sort(key=lambda x:str(x[:-4]))

# Save the predicted image
'''
Note that the mask at T2 is set when the prediction result is output
'''
for i in range(len(result)):
    # Use a mask! The T2 image was masked
    name1 = out_path1 + f_list[i]
    name2 = out_path2 + f_list[i]
    print(name1)
    pred_img = result[i]
    t2_img = t2[i]
    mask = np.where(t2_img>0,1,0)  # To make a mask, the missing part is 0 and the missing part is 1
    img = (1-mask)*pred_img + t2[i]
    #cv.imwrite(name1, img)
    cv.imwrite(name2, pred_img)




