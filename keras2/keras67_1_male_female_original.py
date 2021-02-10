  
# 실습: 남자 여자 구별
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
import random


#1. prepare directories
MALE_DIR = 'C:/data/image/gender/male'
FEMALE_DIR = 'C:/data/image/gender/female'
male_all = os.listdir(MALE_DIR) # 리스트 반환
female_all = os.listdir(FEMALE_DIR)

#2. equalize the length of directories list
DATASET_LEN = min(len(male_all), len(female_all)) # 841 (male_all = 895, female_all = 841)
male_resized = random.sample(male_all, DATASET_LEN) # type == list, size == 841
female_resized = random.sample(female_all, DATASET_LEN)

#3. prepare empty arrays for storing
DIMENSION = 256 ### HYPER-PARAMETER TUNNING 1 ###
CHANNEL = 3
X_male = np.empty((DATASET_LEN, DIMENSION, DIMENSION, CHANNEL))
X_female = np.empty((DATASET_LEN, DIMENSION, DIMENSION, CHANNEL))
y_male = np.empty((DATASET_LEN), dtype=int) # 
y_female = np.empty((DATASET_LEN), dtype=int) # 

#4. access to image files and make it into numpy arrays (X, y)
for i, f in enumerate(male_resized):
    male_img = image.load_img(os.path.join(MALE_DIR, f), target_size=(DIMENSION, DIMENSION))
    x = image.img_to_array(male_img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = np.squeeze(x)
    X_male[i,:,:,:] = x
    y_male[i] = 1

for i, f in enumerate(female_resized):
    female_img = image.load_img(os.path.join(FEMALE_DIR, f), target_size=(DIMENSION, DIMENSION))
    x = image.img_to_array(female_img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = np.squeeze(x)
    X_female[i,:,:,:] = x
    y_female[i] = 0

X = np.concatenate((X_male, X_female), axis=0) # (1682, 64, 64, 3)
y = np.concatenate((y_male, y_female), axis=0) # (1682,)

#5. split data into train and test sets (X_train, X_test, y_train, y_test)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

#6. model
initial_model = VGG16(weights="imagenet", include_top=False, input_shape = (DIMENSION, DIMENSION, 3))
last = initial_model.output

NODE = 4096 ### HYPER-PARAMETER TUNNING 2 ###

x = Flatten()(last)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.4)(x)
preds = Dense(1, activation='sigmoid')(x)
model = Model(initial_model.input, preds)

#7. compile
model.compile(Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

#8. fit
es = EarlyStopping(monitor='val_loss', patience=10)
re = ReduceLROnPlateau(monitor='val_loss', patience=5)

model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.2, callbacks=[es, re])

#9. predict
y_pred = model.predict(X_test)
y_pred = [(y[0]>=0.5).astype(np.uint8) for y in y_pred]

print('DIMENSION, NODE :', DIMENSION, NODE)
print('Accuracy:', np.mean((y_test==y_pred)))

# Reference
# https://www.tensorflow.org/tutorials/load_data/images?hl=ko

# DIMENSION, NODE : 64 4096
# Accuracy: 0.8635014836795252

# DIMENSION, NODE : 128 4096
# Accuracy: 0.8991097922848664

# DIMENSION, NODE : 256 4096
# Accuracy: 0.913946587537092