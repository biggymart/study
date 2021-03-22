# h5 파일 만들기

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


# npy 불러와서 모델 컴파일, 훈련, h5 세이브하는 함수
def mk_h5():
    for i in range(int(len(train_fnames)/atom)):
        x_train = np.load('../data/LPD_competition/npy/Lotte_train_x_{0}.npy'.format(i+1))
        y_train = np.load('../data/LPD_competition/npy/Lotte_train_y_{0}.npy'.format(i+1))
        x_val = np.load('../data/LPD_competition/npy/Lotte_val_x_{0}.npy'.format(i+1))
        y_val = np.load('../data/LPD_competition/npy/Lotte_val_y_{0}.npy'.format(i+1))
        print('npy loaded for', i+1, 'th time')

        initial_model = MobileNetV2(weights="imagenet", include_top=False, input_shape = (DIMENSION, DIMENSION, 3))
        last = initial_model.output

        x = Flatten()(last)
        x = Dense(NODE, activation='relu')(x)
        x = Dropout(DROPOUT_RATE)(x)
        x = Dense(NODE, activation='relu')(x)
        preds = Dense(atom, activation='softmax')(x)
        model = Model(initial_model.input, preds)

        # compile
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # fit
        es = EarlyStopping(monitor='val_loss', patience=10)
        re = ReduceLROnPlateau(monitor='val_loss', patience=5)
        cp = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, mode='auto')

        model.fit(x_train, y_train, epochs=30, validation_data=(x_val, y_val), callbacks=[es, re, cp])
        model.save('../data/LPD_competition/h5/Lotte_model_{0}.h5'.format(i+1))
        print('model saved for', i+1, 'th time')
        gc.collect()


##### 함수 호출 (커맨드센터) #####
if __name__ == "__main__":
    mk_h5()