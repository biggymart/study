import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

########################################
### load data from csv
filepath = 'C:/data/mnist/'
train = pd.read_csv(filepath + 'train.csv') # (2048, 787)
test  = pd.read_csv(filepath + 'test.csv') # (20480, 786)
submission = pd.read_csv(filepath + 'submission.csv') # (20480, 2)
########################################

### 1-0. 전처리
train2 = train.drop(['id', 'digit', 'letter'], axis=1).values
x_test = test.drop(['id', 'letter'], axis=1).values
x_test = x_test.reshape(-1, 28, 28, 1)

# 값이 너무 작은 것은 빼기
x_test = np.where((x_test<=20)&(x_test!=0), 0., x_test) # 조건문과 비슷함, 첫번째 인자 조건 맞으면 두번째를 주고 틀리면 세번째를 줌
x_test = x_test/255. # scaling

y = train['digit']
y_train = np.zeros((len(y), len(y.unique())))  # 총 행의수 , 10(0~9)
for i, digit in enumerate(y): # OneHotEncoding
    y_train[i, digit] = 1 

#이미지 발생기
datagen = ImageDataGenerator(
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.15,
        rotation_range = 10,
        validation_split=0.2)

valgen = ImageDataGenerator()

def create_model():
    model = Sequential()

    model.add(Conv2D(16, (3,3), activation='relu', input_shape=(28, 28,1), padding='same'))
    model.add(BatchNormalization()) 
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization()) 
    model.add(MaxPooling2D(3,3))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='softmax'))
    return model


kfold = KFold(n_splits=5, random_state=40)
for train, val in kfold.split(train2):
    re = ReduceLROnPlateau(patience=5, verbose=1, factor= 0.5)
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=10)
    
    X_train = train2[train]
    X_val = train2[val]
    Y_train = y_train[train]
    Y_val = y_train[val]

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_val = X_val.reshape(-1, 28, 28, 1)

    model = create_model()

    training_generator = datagen.flow(X_train, Y_train, batch_size=32, seed=7, shuffle=True)
    validation_generator = valgen.flow(X_val, Y_val, batch_size=32, seed=7, shuffle=True)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=2e-3, epsilon=None), metrics=['acc'])
    model.fit(training_generator, epochs=1000, callbacks=[es, re], shuffle=True, validation_data=validation_generator)

    results = model.predict_generator(validation_generator)
    print(results)
    print(results.shape)

submission['digit'] = np.argmax(results, axis=1)
submission.to_csv(filepath + 'my_submission.csv', index=False)


'''
# pip install opencv-python
# 유용한 사용법 정리한 블로그 (이미지 읽기, 이미지 컬러 공간 변환, 사이즈 변경, 보여주기, 저장; 그림 그리기, 텍스트 넣기, 채널 분리, 채널 병합, 정규화)
# https://blueskyvision.tistory.com/712
import cv2

메모리 에러 발생시킴 (메모리가 딸리는 듯)
for i, s in enumerate(x_test):
    converted = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB) # Gray -> RGB
    resized = cv2.resize(converted,(300,300),interpolation = cv2.INTER_CUBIC) # INTER_CUBIC 사용 -> 확대
    # cv2.resize(원본 이미지, 결과 이미지 크기, 보간법)
    # 보간법을 이용하여 픽셀들의 값을 할당합니다
    # https://076923.github.io/posts/Python-opencv-8/
    del converted
    test_224[i] = resized
    del resized

# 가비지 콜렉션에 대한 설명
# https://medium.com/dmsfordsm/garbage-collection-in-python-777916fd3189
import gc
from keras import backend as bek
bek.clear_session() # 현재 TF 그래프를 없애고, 새로운 TF 그래프를 만듭니다. 오래된 모델 혹은 층과의 혼란을 피할 때 유용합니다.
gc.collect() # 모든 세대의 가비지 컬렉션을 즉시 수행합니다.

from keras.applications.vgg16 import VGG16
def create_model():
    effnet = VGG16(
        include_top=True,
        weights=None,
        input_shape=(28, 28, 1),
        classes=10,
        classifier_activation="softmax",
    )
    model = Sequential()
    model.add(ZeroPadding2D(padding=(4, 4), data_format=None))
    model.add(effnet)
    model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=2e-3), metrics=['accuracy'])
    return model
'''