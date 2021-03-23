# 데이터 증식해주는 파일

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

##### 변수 및 하이퍼파라미터 조절기 #####
TRAIN_DIR = 'C:/data/LPD_competition/train'
TEST_DIR = 'C:/data/LPD_competition/test'
model_path = 'C:/data/LPD_competition/modelcheckpoint/lotte_0322_1_{epoch:02d}-{val_loss:.4f}.hdf5'

DIMENSION = 128
atom = 100 # 각 모델이 구분할 class의 개수

train_fnames = natsorted(os.listdir(TRAIN_DIR))
test_fnames = natsorted(os.listdir(TEST_DIR))
#######################################

##### 제너레이터 정의 (Initialize image data generator) #####
########################
train_datagen = ImageDataGenerator(
    validation_split = 0.3,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    horizontal_flip=True,
    rotation_range=5,
    zoom_range=0.2,
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
########################