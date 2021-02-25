### Description ###
# 이미지 한 장에 대해서 전처리하도록 해보죠. 잘 되면 더 확장하도록 해요.
# Date: 2021-02-25


### 0. 라이브러리는 여기에 정리하도록 해요
import os
import cv2


### 1. 가장 먼저, 디렉토리를 정리해봅시다!
#1-1. 베이스 디렉토리
train_base_dir = 'C:/data/mnist/dirty_mnist_2nd/'
test_base_dir = 'C:/data/mnist/test_dirty_mnist_2nd/'

#1-2. 파일명 리스트로 정리 (나중에 반복문을 위해)
train_file_lst = os.listdir(train_base_dir)
test_file_lst = os.listdir(test_base_dir)


### 2. 정리한 디렉토리를 바탕으로 이미지를 한 장만 긁어와 볼까요?
img_train = cv2.imread(train_base_dir + train_file_lst[0], flags=cv2.IMREAD_UNCHANGED)
img_test = cv2.imread(test_base_dir + test_file_lst[0], flags=cv2.IMREAD_UNCHANGED)

# 가져온 이미지를 확인해봐요
cv2.imshow('image', img_train)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 





'''
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import matplotlib.pyplot as plt
import os
import random

from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import Model
import numpy as np
import random
import os
import matplotlib.pyplot as plt

TRAIN_DIR = 'C:\data\mnist\dirty_mnist_2nd'
f_train, f_valid = train_test_split(os.listdir(TRAIN_DIR), test_size=0.7, random_state=42)

PATCH_DIM = 32

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, files, batch_size=32, dim=(224,224), n_channels=3, n_classes=2, shuffle=True):
        self.files = files
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle

    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = np.random.choice(len(self.files), self.batch_size)
        
        # Find files of IDs
        batch_files = self.files[indexes]

        # Generate data
        X, y = self.__data_generation(batch_files)

        return X, y

    def __data_generation(self, batch_files):
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, f in enumerate(batch_files):
            # Store sample
            img_path = os.path.join(TRAIN_DIR,f)
            img = image.load_img(img_path, target_size=self.dim)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            x = np.squeeze(x)
            X[i,:,:,:] = x

            # Store class
            if 'dog' in f:
                y[i]=1
            else:
                y[i]=0
                
        return X, y

training_generator = DataGenerator(np.array(f_train), dim=(PATCH_DIM, PATCH_DIM))


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(strides=2))
model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(MaxPool2D(strides=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))
'''

### 참고문헌
# 딥러닝과 OpenCV를 활용하여 사진 속 글자 검출하기
# https://d2.naver.com/helloworld/8344782