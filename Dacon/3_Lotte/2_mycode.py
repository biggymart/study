import numpy as np
import pandas as pd
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
import csv

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

test_datagen = ImageDataGenerator(rescale=1./255)

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
preds = Dense(1000, activation='softmax')(x)
model = Model(initial_model.input, preds)

#7. compile
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['acc'])

#8. fit
es = EarlyStopping(monitor='val_loss', patience=10)
re = ReduceLROnPlateau(monitor='val_loss', patience=5)
hist = model.fit_generator(xy_train,
        steps_per_epoch=10, epochs= 100, callbacks=[es, re],
        validation_data=xy_val)


f = open('C:/Users/ai/Downloads/LPD_competition/sample.csv','r')
rdr = csv.reader(f)
lines = []
for line in rdr:
    if line[1] == 'prediction':
        line[1] = 'prediction'
        lines.append(line)
    else:
        line[1] = model.predict()
        lines.append(line)
 
f = open('Random.csv','w',newline='') #원본을 훼손할 위험이 있으니 다른 파일에 저장하는 것을 추천합니다.
wr = csv.writer(f)
wr.writerows(lines)
 
f.close()


result = model.predict(test_generator,verbose=True)
sub = pd.read_csv('../../data/image/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('../../data/image/answer3.csv',index=False)