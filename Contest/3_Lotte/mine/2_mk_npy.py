# npy 만들자

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

# 1000/X개 모델을 훈련할 데이터 각 X개 클래스로 구성해서 npy로 저장하는 함수
def mk_npy(atom):
    for i in range(int(len(train_fnames)/atom)): # 1000/X 만큼 반복
        xy_train = train_datagen.flow_from_directory(
            directory=TRAIN_DIR,
            class_mode='categorical',
            classes=train_fnames[i:i+atom],
            subset='training',
            target_size=(DIMENSION, DIMENSION)
        )
        xy_val = train_datagen.flow_from_directory(
            directory=TRAIN_DIR,
            class_mode='categorical',
            classes=train_fnames[i:i+atom],
            subset='validation',
            target_size=(DIMENSION, DIMENSION)
        )
        np.save('../data/LPD_competition/npy/Lotte_train_x_{0}.npy'.format(i+1), arr=xy_train[0][0])
        np.save('../data/LPD_competition/npy/Lotte_train_y_{0}.npy'.format(i+1), arr=xy_train[0][1])
        np.save('../data/LPD_competition/npy/Lotte_val_x_{0}.npy'.format(i+1), arr=xy_val[0][0])
        np.save('../data/LPD_competition/npy/Lotte_val_y_{0}.npy'.format(i+1), arr=xy_val[0][1])
        print('npy files saved for ', i+1, 'th time')


##### 함수 호출 (커맨드센터) #####
if __name__ == "__main__":
    test_atom(atom)
    mk_npy(atom)