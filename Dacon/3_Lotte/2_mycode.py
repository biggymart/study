import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

### HYPER-PARAMETER TUNNING ###
DIMENSION = 256
ROUNDS = 1
NODE = 4096

TRAIN_DIR = 'C:/Users/ai/Downloads/LPD_competition/train'

datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=20,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest',
    validation_split=0.2
)

xy_train = datagen.flow_from_directory(
    TRAIN_DIR, 
    target_size=(DIMENSION, DIMENSION),
    color_mode='rgb',
    batch_size=10,
    class_mode='categorical',
    subset="training"
) # Found 48000 images belonging to 1000 classes.

xy_val = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(DIMENSION, DIMENSION),
    color_mode='rgb',
    batch_size=10,
    class_mode='categorical',
    subset="validation"
)

# np.save('C:/Users/ai/Downloads/LPD_competition/npy/LG_x.npy', arr=xy_train[0][0])
# np.save('C:/Users/ai/Downloads/LPD_competition/npy/LG_y.npy', arr=xy_train[0][1])

# x = np.load('C:/Users/ai/Downloads/LPD_competition/npy/LG_x.npy')
# y = np.load('C:/Users/ai/Downloads/LPD_competition/npy/LG_y.npy')

initial_model = VGG16(weights="imagenet", include_top=False, input_shape=(DIMENSION, DIMENSION, 3))
last = initial_model.output

x = Flatten()(last)
x = Dense(NODE, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(NODE, activation='relu')(x)
x = Dropout(0.4)(x)
preds = Dense(1, activation='sigmoid')(x)
model = Model(initial_model.input, preds)

#7. compile
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['acc'])

#8. fit
es = EarlyStopping(monitor='val_loss', patience=10)
re = ReduceLROnPlateau(monitor='val_loss', patience=5)
hist = model.fit_generator(xy_train,
        steps_per_epoch=10, epochs= 100, callbacks=[es, re],
        validation_data=xy_val)
