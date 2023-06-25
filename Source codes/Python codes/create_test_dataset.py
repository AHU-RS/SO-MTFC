import os
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from shutil import copyfile
import shutil

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

        # Read the data dimension(24,24)
        t1_img = cv.imread(t1_name, cv.IMREAD_UNCHANGED)
        t2_img = cv.imread(t2_name, cv.IMREAD_UNCHANGED)
        t3_img = cv.imread(t3_name, cv.IMREAD_UNCHANGED)
        label_img = cv.imread(label_name, cv.IMREAD_UNCHANGED)
        t1_img = t1_img.reshape([img_size, img_size, 1])
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

def get_filename(path,suffix):
    """Gets the file with the specified suffix in the specified directory"""
    file_list = []
    f_list = os.listdir(path)
    f_list.sort(key=lambda x:str(x[:-4]))  # .tif   Notice that we use int to sort numbers here; Others are sorted by str
    for file in f_list:
        if os.path.splitext(file)[1] == suffix:
            file_list.append(os.path.join(path,file))
    return file_list

def test_data():
    train_path = 'F:/Day_S0/dataset1/train2/'
    test_path = 'F:/Day_S0/dataset1/test2/'

    input1_path = train_path+'/t1/'
    input2_path = train_path+'/t2/'
    input3_path = train_path+'/t3/'
    input_label_path = train_path+'/label/'


    target_T1_path = test_path+'/t1/'
    target_T2_path = test_path+'/t2/'
    target_T3_path = test_path+'/t3/'
    target_label_path = test_path+'/label/'

    if os.path.exists(target_T1_path):
        shutil.rmtree(target_T1_path)
    os.makedirs(target_T1_path)

    if os.path.exists(target_T2_path):
        shutil.rmtree(target_T2_path)
    os.makedirs(target_T2_path)

    if os.path.exists(target_T3_path):
        shutil.rmtree(target_T3_path)
    os.makedirs(target_T3_path)

    if os.path.exists(target_label_path):
        shutil.rmtree(target_label_path)
    os.makedirs(target_label_path)

    t1_names = os.listdir(input1_path)
    t2_names = os.listdir(input2_path)
    t3_names = os.listdir(input3_path)
    tlabel = os.listdir(input_label_path)

    sum = len(t1_names)
    n = 0

    t1_names.sort(key=lambda x: str(x[:-4]))
    t2_names.sort(key=lambda x: str(x[:-4]))
    t3_names.sort(key=lambda x: str(x[:-4]))
    tlabel.sort(key=lambda x: str(x[:-4]))
    # Data shuffling
    np.random.seed(3700)
    np.random.shuffle(t1_names)
    np.random.seed(3700)
    np.random.shuffle(t2_names)
    np.random.seed(3700)
    np.random.shuffle(t3_names)
    np.random.seed(3700)
    np.random.shuffle(tlabel)
    np.random.seed(3700)

    for i in range(0,sum,9):
        t1_name = t1_names[i]
        t2_name = t2_names[i]
        t3_name = t3_names[i]
        tlabel_name = tlabel[i]

        print(n+1,t1_name,'\t',t2_name,'\t',t3_name,'\t',tlabel_name)
        n=n+1

        shutil.move(input1_path+t1_name,target_T1_path+t1_name)
        shutil.move(input2_path+t2_name,target_T2_path+t2_name)
        shutil.move(input3_path+t3_name,target_T3_path+t3_name)
        shutil.move(input_label_path+tlabel_name,target_label_path+tlabel_name)

if __name__ == '__main__':
    test_data()
