# https://github.com/sassoftware/python-dlpy/blob/master/examples/image_classification/Image_Classification_Fruits_With_EfficientNet.ipynb
# 위 코드를 보고 해보자

# 비지도 다중분류 -> OneHotEncoding, softmax

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import numpy as np

TRAIN_DIR = 'C:/data/image/beer'
beer_names = os.listdir(TRAIN_DIR)

PATCH_DIM = 150
images_num = 1250
batch_num = 8
steps_num = int(images_num / batch_num)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest', 
    validation_split=0.2  
)

xy_train = train_datagen.flow_from_directory(
    TRAIN_DIR, 
    target_size=(PATCH_DIM, PATCH_DIM),
    batch_size=batch_num, 
    class_mode='categorical',
    subset='training'
)

xy_val = train_datagen.flow_from_directory(
    TRAIN_DIR, 
    target_size=(PATCH_DIM, PATCH_DIM),
    batch_size=batch_num, 
    class_mode='categorical',
    subset='validation'
)

# Resnet도 해야지
initial_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(PATCH_DIM, PATCH_DIM, 3)
)
last = initial_model.output

x = Flatten()(last)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
preds = Dense(4, activation='softmax')(x)

model = Model(initial_model.input, preds)

model.compile(loss='categorical_crossentropy', 
            optimizer='adam',
            metrics=['acc']
)

re = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.2)
es = EarlyStopping(monitor='val_loss', patience=10)
hist = model.fit_generator(
    xy_train,
    steps_per_epoch= xy_train.samples // batch_num,
    epochs=100,
    validation_data= xy_val,
    validation_steps= xy_val.samples // batch_num,
    callbacks=[es, re]
)

model.save('C:/data/h5/train.h5')
acc = hist.history['acc']
loss = hist.history['loss']

print("acc :", acc[-1])

#5. Visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6)) # 도화지 면적을 잡아줌, 가로가 10, 세로가 6

plt.subplot(2, 1, 1) # 2행 1열 짜리 그래프를 만들겠다, 그 중 첫번째
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid() # 격자

plt.title('Cost Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2) # 2행 1열 그래프 중 두번째
plt.plot(hist.history['acc'], marker='.', c='red', label='accuracy')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_loss')
plt.grid()

plt.title('Accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()

# Epoch 46/1000
# 125/125 [==============================] - 10s 82ms/step - loss: 0.2492 - acc: 0.9090 - val_loss: 0.4665 - val_acc: 0.8105
# acc : 0.9089999794960022

# Epoch 38/100
# 125/125 [==============================] - 10s 83ms/step - loss: 0.6262 - acc: 0.7350 - val_loss: 0.8366 - val_acc: 0.6371
# acc : 0.7350000143051147