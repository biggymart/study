# https://keras.io/ko/preprocessing/image/

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 데이터 제너레이터 선언
train_datagen = ImageDataGenerator(
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

test_datagen = ImageDataGenerator(rescale=1./255)

# 제너레이터로부터 데이터 만들기
xy_train = train_datagen.flow_from_directory(
    '../data/image/brain/train', 
    target_size=(150, 150),
    batch_size=5, 
    class_mode='binary'
    , save_to_dir='../data/image/brain_generator/train/'
) # Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '../data/image/brain/test',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary'
    , save_to_dir='../data/image/brain_generator/test/'
) # Found 120 images belonging to 2 classes.

print(xy_train[0][0])
# 정의한 변수에 액션을 줘야 save_to_dir이 작동함
# 증폭되는 사진의 갯수 = (batch_size 갯수에 정의한 갯수) * (건드린 횟수)
