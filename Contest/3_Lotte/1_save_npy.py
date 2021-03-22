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
DIMENSION = 128
BATCH = 20
NODE = 1024

TRAIN_DIR = 'C:/data/LPD_competition/train'
TEST_DIR = 'C:/data/LPD_competition/test'
model_path = 'C:/data/LPD_competition/modelcheckpoint/lotte_0318_1_{epoch:02d}-{val_loss:.4f}.hdf5'

TRAINING_SIZE = 39000
VALIDATION_SIZE = 9000
TEST_SIZE = 72000
##############################

### 1. data
# generator declaration
train_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=20,
    zoom_range=0.1,
    validation_split=0.2,
    preprocessing_function=preprocess_input
)

test_gen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=preprocess_input
) 

# generator flow
train_xy = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(DIMENSION, DIMENSION),
    batch_size=BATCH,
    class_mode='categorical',
    subset="training"
) # Found 39000 images belonging to 1000 classes.

val_xy = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(DIMENSION, DIMENSION),
    batch_size=BATCH,
    class_mode='categorical',
    subset="validation"
) # Found 9000 images belonging to 1000 classes.

# prepare empty arrays for storing
test_files = os.listdir(TEST_DIR)
test_x = np.empty((TEST_SIZE, DIMENSION, DIMENSION, 3))
#==== until here, only dir ====#

# access to image files and make it into a numpy array (X, y)
for i, f in enumerate(test_files):
    # ver1
    # img = image.load_img(os.path.join(TEST_DIR, f), target_size=(DIMENSION, DIMENSION)) # access to folder and load as image
    # x = np.expand_dims(img, axis=0) # flow 위해서 4D tensor로 일단 만들어준다
    # test_x = test_gen.flow(x)

    # ver2
    img = image.load_img(os.path.join(TEST_DIR, f), target_size=(DIMENSION, DIMENSION)) # access to folder and load as image
    x = image.img_to_array(img) # now dtype is array
    x = np.expand_dims(x, axis=0) # MobileNet 전처리 위해서 4D tensor로 일단 만들어준다
    x = preprocess_input(x)
    x = np.squeeze(x) # 전처리했으니 다시 3D tensor로 만들어준다
    test_x[i,:,:,:] = x

np.save('C:/data/LPD_competition/npy/L_train_x.npy', arr=train_xy[0][0])
np.save('C:/data/LPD_competition/npy/L_train_y.npy', arr=train_xy[0][1])
np.save('C:/data/LPD_competition/npy/L_val_x.npy', arr=val_xy[0][0])
np.save('C:/data/LPD_competition/npy/L_val_y.npy', arr=val_xy[0][1])
np.save('C:/data/LPD_competition/npy/L_test_x.npy', arr=test_x)
