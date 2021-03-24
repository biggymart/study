# 1000개 그냥 다 싹다 전이학습 해야지

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

train_fnames = natsorted(os.listdir(TRAIN_DIR))
test_fnames = natsorted(os.listdir(TEST_DIR))

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


# 1000개 싹다 불러와서 npy로 저장
def mk_npy():
    xy_train = train_datagen.flow_from_directory(
        directory=TRAIN_DIR,
        class_mode='categorical',
        subset='training',
        target_size=(DIMENSION, DIMENSION)
    )
    xy_val = train_datagen.flow_from_directory(
        directory=TRAIN_DIR,
        class_mode='categorical',
        subset='validation',
        target_size=(DIMENSION, DIMENSION)
    )
    np.save('../data/LPD_competition/npy/Lotte_train_x_whole.npy', arr=xy_train[0][0])
    np.save('../data/LPD_competition/npy/Lotte_train_y_whole.npy', arr=xy_train[0][1])
    np.save('../data/LPD_competition/npy/Lotte_val_x_whole.npy', arr=xy_val[0][0])
    np.save('../data/LPD_competition/npy/Lotte_val_y_whole.npy', arr=xy_val[0][1])
    gc.collect()
    print("*** npy saved ***")

# npy 불러와서 모델 컴파일, 훈련, h5 세이브하는 함수
def mk_h5():
    x_train = np.load('../data/LPD_competition/npy/Lotte_train_x_whole.npy')
    y_train = np.load('../data/LPD_competition/npy/Lotte_train_y_whole.npy')
    x_val = np.load('../data/LPD_competition/npy/Lotte_val_x_whole.npy')
    y_val = np.load('../data/LPD_competition/npy/Lotte_val_y_whole.npy')

    initial_model = MobileNetV2(weights='imagenet', include_top=False, input_shape = (DIMENSION, DIMENSION, 3))
    # initial_model.trainable = True
    last = initial_model.output

    x = Flatten()(last)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    preds = Dense(1000, activation='softmax')(x)
    model = Model(initial_model.input, preds)

    # compile
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # fit
    es = EarlyStopping(monitor='val_loss', patience=15)
    re = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.3, verbose=1)
    cp = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, mode='auto')

    model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), callbacks=[es, re, cp])
    model.save('C:/data/LPD_competition/h5/Lotte_model_whole.h5')
    print('*** model saved ***')
    gc.collect()

def predict(image_path):
    h5_dir = 'C:/data/LPD_competition/h5'
    
    # 10개의 값이 들어간 리스트가 되어야
    pred_lst = []

    # 모델에 데이터 shape 맞도록 전처리
    test_img = image.load_img(image_path, target_size=(DIMENSION, DIMENSION))
    x = image.img_to_array(test_img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # 모델 불러옴
    model = load_model('../data/LPD_competition/h5/Lotte_model_whole.h5')

    y_pred = model.predict(x)
    # print("y_pred의 shape :", y_pred.shape)

    pred_lst.append(np.argmax(y_pred))

    return pred_lst




    
##### 함수 호출 (커맨드센터) #####
if __name__ == "__main__":
    mk_npy()
    mk_h5()
    
    empty_lst = []
    sub = pd.DataFrame(columns=['filename', 'prediction'])
    for idx, img in enumerate(test_fnames): # test folder --> 72000 imgs
        # if idx >= 5: # 이미지 X개만 해볼까
        #     break
        gc.collect()
        tf.keras.backend.clear_session()

        print("********", idx+1, "th img being processed...", "********")
        base_dir = TEST_DIR + '/' # 각 폴더 디렉토리 'C:/data/LPD_competition/test/'
        img_lst = natsorted(os.listdir(base_dir)) # 각 폴더 안에 있는 jpg 파일 리스트
        img_dir = base_dir + img # 'C:/data/LPD_competition/train/0/0.jpg'
        
        aaa = predict(img_dir)
        sub = sub.append({'filename': "{0}.jpg".format(idx+1), 'prediction': aaa[0]}, ignore_index=True)
        sub.to_csv('C:/data/LPD_competition/lotte0323_1.csv',index=False)
        
        gc.collect()
        tf.keras.backend.clear_session()
