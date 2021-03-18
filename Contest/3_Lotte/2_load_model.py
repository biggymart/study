import numpy as np
import pandas as pd
import os
import csv
import math
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

### HYPER-PARAMETER TUNNING ###
DIMENSION = 64
BATCH = 128
NODE = 1024

TRAIN_DIR = 'C:/data/LPD_competition/train'
TEST_DIR = 'C:/data/LPD_competition/test'
# model_path = 'C:/data/modelCheckpoint/lotte_0318_1_{epoch:02d}-{val_loss:.4f}.hdf5'

TRAINING_SIZE = 39000
VALIDATION_SIZE = 9000
TEST_SIZE = 72000
##############################

train_x = np.load('C:/data/LPD_competition/npy/L_train_x.npy')
train_y = np.load('C:/data/LPD_competition/npy/L_train_y.npy')
val_x = np.load('C:/data/LPD_competition/npy/L_val_x.npy')
val_y = np.load('C:/data/LPD_competition/npy/L_val_y.npy')
test_x = np.load('C:/data/LPD_competition/npy/L_test_x.npy')

### 2. modeling
initial_model = MobileNetV2(
     include_top=False,
     input_shape=(DIMENSION, DIMENSION, 3)
)
last = initial_model.output

x = Flatten()(last)
x = Dropout(0.2)(x)
x = Dense(NODE, activation='relu', kernel_initializer = 'he_normal')(x)
x = Dense(NODE, activation='relu', kernel_initializer = 'he_normal')(x)
preds = Dense(1000, activation='softmax')(x)
model = Model(initial_model.input, preds)

### 3. compile
model.compile(optimizer='adam',
     loss='categorical_crossentropy',
    metrics=['acc'])

### 4. fit
compute_steps_per_epoch = lambda x: int(math.ceil(1. * x / BATCH))
es = EarlyStopping(monitor='val_loss', patience=5)
re = ReduceLROnPlateau(monitor='val_loss', patience=3)
# cp = ModelCheckpoint(model_path, save_best_only= True)
hist = model.fit(train_x, train_y,
        steps_per_epoch=compute_steps_per_epoch(TRAINING_SIZE),
        epochs= 10, callbacks=[es, re],
        validation_data=(val_x, val_y),
        validation_steps=compute_steps_per_epoch(VALIDATION_SIZE)
)

# model = load_model(모델경로넣어라)

pred = model.predict(test_x)
pred = np.argmax(pred, 1)
sub = pd.read_csv('C:/data/LPD_competition/sample.csv', header = 0)
sub.loc[:,'prediction'] = pred
sub.to_csv('C:/data/LPD_competition/csv/lotte01.csv', index = False)
