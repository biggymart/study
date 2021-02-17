from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import csv

TRAIN_DIR = 'C:/data/image/beer'

PATCH_DIM = 150
images_num = 82
batch_num = 1
steps_num = images_num / batch_num

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)

xy_train = train_datagen.flow_from_directory(
    'C:/data/image/beer', 
    target_size=(150, 150),
    batch_size=batch_num, 
    class_mode='binary'
) # 1650 images

# Resnet도 해야지
initial_model = VGG16(weights="imagenet", include_top=False, input_shape = (PATCH_DIM, PATCH_DIM, 3))
last = initial_model.output

x = Flatten()(last)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
preds = Dense(1, activation='sigmoid')(x)

model = Model(initial_model.input, preds)

model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['acc'])

es = EarlyStopping(monitor='loss', patience=5)
hist = model.fit_generator(
    xy_train, steps_per_epoch=steps_num, epochs=100, callbacks=[es]
)

acc = hist.history['acc']
loss = hist.history['loss']

print("acc :", acc[-1])

#5. Visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6)) # 도화지 면적을 잡아줌, 가로가 10, 세로가 6

plt.subplot(2, 1, 1) # 2행 1열 짜리 그래프를 만들겠다, 그 중 첫번째
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.grid() # 격자

plt.title('Cost Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2) # 2행 1열 그래프 중 두번째
plt.plot(hist.history['acc'], marker='.', c='red', label='accuracy')
plt.grid()

plt.title('Accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()