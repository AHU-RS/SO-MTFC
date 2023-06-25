import numpy as np
from utils import *
from model import *
from sklearn.metrics import r2_score
t1_path = 'E:/sy_data/AMSR-E_night_data/Night-2010-S1/01-31/dataset1h/temporal1/'
t2_path = 'E:/sy_data/AMSR-E_night_data/Night-2010-S1/01-31/dataset1h/temporal2/'
t3_path = 'E:/sy_data/AMSR-E_night_data/Night-2010-S1/01-31/dataset1h/temporal3/'
label_path = 'E:/sy_data/AMSR-E_night_data/Night-2010-S1/01-31/dataset1h/origin2/'


def test_data():
    input1_path = 'F:/Day_S3/dataset1/t1/'
    input2_path = 'F:/Day_S3/dataset1/t2/'
    input3_path = 'F:/Day_S3/dataset1/t3/'
    input_label_path = 'F:/Day_S3/dataset1/t2_label/'


    target_T1_path = 'F:/Day_S3/test/t1/'
    target_T2_path = 'F:/Day_S3/test/t2/'
    target_T3_path = 'F:/Day_S3/test/t3/'
    target_label_path = 'F:/Day_S3/test/t2_label/'



    t1_names = os.listdir(input1_path)
    t2_names = os.listdir(input2_path)
    t3_names = os.listdir(input3_path)
    tlabel = os.listdir(input_label_path)


    sum = len(t1_names)
    n = 0
    #
    t1_names.sort(key=lambda x: str(x[:-4]))
    t2_names.sort(key=lambda x: str(x[:-4]))
    t3_names.sort(key=lambda x: str(x[:-4]))
    tlabel.sort(key=lambda x: str(x[:-4]))


    # Data shuffling
    np.random.seed(37)
    np.random.shuffle(t1_names)
    np.random.seed(37)
    np.random.shuffle(t2_names)
    np.random.seed(37)
    np.random.shuffle(t3_names)
    np.random.seed(37)
    np.random.shuffle(tlabel)
    np.random.seed(37)

    for i in range(0,sum,10):
        t1_name = t1_names[i]
        t2_name = t2_names[i]
        t3_name = t3_names[i]
        tlabel_name = tlabel[i]

        print(n+1,t1_name,'\t',t2_name,'\t',t3_name,'\t',tlabel_name)
        n=n+1


        # shutil.move(input1_path+t1_name,target_T1_path+t1_name)
        # shutil.move(input2_path+t2_name,target_T2_path+t2_name)
        # shutil.move(input3_path+t3_name,target_T3_path+t3_name)
        # shutil.move(input_label_path+tlabel_name,target_label_path+tlabel_name)


        copyfile(input1_path+t1_name,target_T1_path+t1_name)
        copyfile(input2_path+t2_name,target_T2_path+t2_name)
        copyfile(input3_path+t3_name,target_T3_path+t3_name)
        copyfile(input_label_path+tlabel_name,target_label_path+tlabel_name)


test_data()