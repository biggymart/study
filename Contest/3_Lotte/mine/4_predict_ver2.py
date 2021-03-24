# 10개의 판별 모델을 만들어서 competition 벌이는 대환장 소스코드!
# 빈 데이터프레임 만들어서 거기에 하나하나 append하는 방식

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from natsort import natsorted
import tensorflow as tf
import pandas as pd
import cv2 as cv
import numpy as np
import os
import gc

import warnings
warnings.filterwarnings(action="ignore")

##### 변수 및 하이퍼파라미터 조절기 #####
TRAIN_DIR = 'C:/data/LPD_competition/train'
TEST_DIR = 'C:/data/LPD_competition/test'
model_path = 'C:/data/LPD_competition/modelcheckpoint/lotte_0322_1_{epoch:02d}-{val_loss:.4f}.hdf5'

DIMENSION = 128
atom = 100 # 각 모델이 구분할 class의 개수

train_fnames = natsorted(os.listdir(TRAIN_DIR))
test_fnames = natsorted(os.listdir(TEST_DIR))

NODE = 4096
DROPOUT_RATE = 0.2
#######################################

##### 제너레이터 정의 (Initialize image data generator) #####
########################
train_datagen = ImageDataGenerator(
    validation_split = 0.2,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    horizontal_flip=True,
    rotation_range=5,
    zoom_range=0.2,
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
########################


##### 함수 호출 (커맨드센터) #####
if __name__ == "__main__":
    sub = pd.DataFrame(columns=['filename', 'prediction'])
    for idx, img in enumerate(natsorted(test_fnames)): # test folder --> 72000 imgs
        # if idx >= 2: # 이미지 X개만 해볼까
        #     break

        print("********", idx+1, "th img being processed...", "********")
        base_dir = TEST_DIR + '/' # 각 폴더 디렉토리 'C:/data/LPD_competition/test/'
        # img_lst = natsorted(os.listdir(base_dir)) # 각 폴더 안에 있는 jpg 파일 리스트
        img_dir = base_dir + img # 'C:/data/LPD_competition/train/0/0.jpg'
        
        # predict(img_dir)
        h5_dir = 'C:/data/LPD_competition/h5'
        h5_lst = os.listdir(h5_dir) # ten h5 models
        
        # 10개의 값이 들어간 리스트가 되어야
        survival = [] # raw value list
        pred_lst = [] # processed value list

        
        for i, f in enumerate(h5_lst): # for each h5
            # if i >= 1:
            #     break

            # 모델에 데이터 shape 맞도록 전처리
            test_img = image.load_img(img_dir, target_size=(DIMENSION, DIMENSION))
            x = image.img_to_array(test_img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            gc.collect()
            tf.keras.backend.clear_session()


            # 모델 하나씩 불러옴
            model = load_model('../data/LPD_competition/h5/Lotte_model_{0}.h5'.format(i+1))
            y_pred = model.predict(x)
            # print("y_pred의 shape :", y_pred.shape)
            # print(i+1, "번째 모델의 인덱스는", np.argmax(y_pred),\
                # "\t그 예측 정확도는 ", y_pred[0, np.argmax(y_pred)]) # raw softmax value
            # print("append한 값 :", (i * atom) + np.argmax(y_pred)) 
            survival.append(y_pred[0, np.argmax(y_pred)]) # raw softmax value
            pred_lst.append((i * atom) + np.argmax(y_pred)) # processed value

            gc.collect()
            tf.keras.backend.clear_session()
        print(survival)
        print(pred_lst)

        final_alive = np.argmax(survival)
        print(final_alive)

        # sub = pd.DataFrame(columns=['filename', 'prediction'])
        sub = sub.append({'filename': "{0}.jpg".format(idx+1), 'prediction': pred_lst[final_alive]}, ignore_index=True)
        sub.to_csv('C:/data/LPD_competition/lotte0324_1.csv',index=False)

    # https://www.tensorflow.org/api_docs/python/tf/keras/backend/clear_session