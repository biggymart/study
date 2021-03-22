# 데이터 증식해주는 파일

from .var_parameters import TRAIN_DIR, TEST_DIR, model_path, DIMENSION, atom, train_fnames, test_fnames, NODE, DROPOUT_RATE, train_datagen, test_datagen

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from natsort import natsorted
import pandas as pd
import cv2 as cv
import numpy as np
import os
import gc

# 1000의 약수인지 확인해주는 함수
def test_atom(atom):
    if len(train_fnames) % atom == 0:
        print("atom well set!")
    else: 
        raise ValueError('atom should evenly divide 1000')

# 각 폴더에 있는 사진을 뻥튀기해줌
def augment_img(multiply_by=2): # 아무 값 안 주면 2배로 뿔려줌
    for idx, folder in enumerate(train_fnames): # 트레인 폴더는 1000개 있다
        # if idx >= 1: # 폴더 X개만 해볼까
        #     break

        base_dir = TRAIN_DIR + '/' + folder + '/' # 각 폴더 디렉토리 'C:/data/LPD_competition/train/0'
        img_lst = natsorted(os.listdir(base_dir)) # 각 폴더 안에 있는 jpg 파일 리스트

        for i, f in enumerate(img_lst): # 각 폴더에 접근해서
            # if i >= 1: # 이미지 X개만 해볼까
            #     break

            img_dir = base_dir + f # 'C:/data/LPD_competition/train/0/0.jpg'
            img = np.expand_dims(image.load_img(img_dir, target_size=(DIMENSION, DIMENSION)), axis=0) # 각 이미지를 불러온다

            ### 눈으로 확인하고 싶으면 uncomment
            # cv_img = cv.imread(img_dir)
            # cv.imshow("whatever", cv_img)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            
            train_datagen.fit(img)
            for x, val in zip(
                train_datagen.flow(x=img,
                save_to_dir=base_dir, # this is where we figure out where to save
                save_prefix='aug', # it will save the images as 'aug_0912' some number for every new augmented image
                shuffle=False
                ), range(multiply_by)) : # here we define a range because we want 10 augmented images otherwise it will keep looping forever I think
                pass # 출처: https://stackoverflow.com/questions/47826730/how-to-save-resized-images-using-imagedatagenerator-and-flow-from-directory-in-k
            print("base dir :", base_dir, "finished ", x)



##### 함수 호출 (커맨드센터) #####
if __name__ == "__main__":
    test_atom(atom)
    augment_img(multiply_by=5)