# ImageDataGenerator를 적용해서 해봤는데
# 배치사이즈도 안 맞고 (버그 있음), 정확도도 떨어짐
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import load_model
import random

### HYPER-PARAMETER TUNNING ###
DIMENSION = 256
CHANNEL = 3 # RGB
ROUNDS = 1
NODE = 4096

#########################################################################################
#1. prepare directories
MALE_DIR = 'C:/data/image/gender/male'
FEMALE_DIR = 'C:/data/image/gender/female'
male_all = os.listdir(MALE_DIR) # returns a list of filenames 
female_all = os.listdir(FEMALE_DIR)

#2. equalize the length of directories list
# (나중에 해당 폴더에 사진 크롤링하면 데이터 보강 가능)
DATASET_LEN = min(len(male_all), len(female_all)) # 841 (male_all = 895 files, female_all = 841 files)
male_resized = random.sample(male_all, DATASET_LEN) # type == list, size == 841
female_resized = random.sample(female_all, DATASET_LEN)

#3. prepare empty arrays for storing
X_male = np.empty((DATASET_LEN, DIMENSION, DIMENSION, CHANNEL))
X_female = np.empty((DATASET_LEN, DIMENSION, DIMENSION, CHANNEL))
y_male = np.empty((DATASET_LEN), dtype=int)
y_female = np.empty((DATASET_LEN), dtype=int)
#==== until here, only dir ====#


#4. access to image files and make it into numpy arrays (X, y)
for i, f in enumerate(male_resized):
    male_img = image.load_img(os.path.join(MALE_DIR, f), target_size=(DIMENSION, DIMENSION)) # access to folder and load as image
    x = image.img_to_array(male_img) # now dtype is array
    x = np.expand_dims(x, axis=0) # VGG16 전처리 위해서 4D tensor로 일단 만들어준다
    x = preprocess_input(x)
    x = np.squeeze(x) # 전처리했으니 다시 3D tensor로 만들어준다
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

_, X_val, _, y_val = train_test_split(X_train, y_train, train_size=0.8, shuffle=True)

####################
# ImageDataGenerator 이용해서 이미지 증폭 -> variable에 리스트로 저장하고 ->  concatenate
# male 과 female에 대해서 각각 <반복문 + file + y_new 에 0, 1 표기>
# datagen은 한 개, 반복문은 두 개
datagen = image.ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest' # 최근접기법
)
datagen2 = image.ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest' # 최근접기법
)
# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
datagen.fit(X_train, augment=True, rounds=ROUNDS, seed=1)
datagen2.fit(X_val, augment=True, rounds=ROUNDS, seed=1)
AUG_DATASET_LEN = DATASET_LEN * ROUNDS
####################

# save and load data
np.save('../data/npy/k67_1_male_female_X_train.npy', arr=X_train)
np.save('../data/npy/k67_1_male_female_X_test.npy', arr=X_test)
np.save('../data/npy/k67_1_male_female_y_train.npy', arr=y_train)
np.save('../data/npy/k67_1_male_female_y_test.npy', arr=y_test)
# X_train = np.load('../data/npy/k67_1_male_female_X_train.npy')
# X_test = np.load('../data/npy/k67_1_male_female_X_test.npy')
# y_train = np.load('../data/npy/k67_1_male_female_y_train.npy')
# y_test = np.load('../data/npy/k67_1_male_female_y_test.npy')
#########################################################################################


#6. model (VGG16 전이학습)
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
hist = model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
        steps_per_epoch=AUG_DATASET_LEN / 32, epochs= 100, callbacks=[es, re],
        validation_data=datagen2.flow(X_val, y_val, batch_size=32))
# validation_split=0.2
# ValueError: `validation_split` is only supported for Tensors or NumPy arrays

# visualization
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])

# plt.title('loss & acc')
# plt.ylabel('loss, acc')
# plt.xlabel('epoch')
# plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])
# plt.show()

# save and load model
model.save('../data/h5/k67_1_male_female.h5')
# model = load_model('../data/h5/k67_1_male_female.h5')
#########################################################################################


# evaluate
loss, acc = model.evaluate(X_test, y_test, batch_size=32)
print("DIMENSION :", DIMENSION, "\tCHANNEL :", CHANNEL, "\tROUNDS :", ROUNDS, "\tNODE", NODE)
print("loss :", loss)
print("acc :", acc)

# predict
y_pred = model.predict(X_test) # 마지막 레이어 sigmoid 함수를 통해 0 ~ 1 사이의 값을 반환, shape==(337, 1)
#[[4.47866887e-06]
# [3.17825680e-03] (이하 생략)

y_pred = [(y[0]>=0.5).astype(np.uint8) for y in y_pred]
# 반복문 한 줄로 해서 리스트 형식으로 변수에 저장
# 0.5보다 크면 1 (True), 작으면 0 (False), unit8 "unsigned 8bit int"


'''
# DIMENSION : 128         CHANNEL : 3     ROUNDS : 1      NODE 4096
# loss : 6.708610534667969
# acc : 0.8635014891624451

DIMENSION : 256         CHANNEL : 3     ROUNDS : 1      NODE 4096
loss : 4.828266143798828
acc : 0.8516320586204529
'''