  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, KFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input


#데이터 지정 및 전처리
x = np.load("../data/LPD_competition/npy/Lotte_train_x5.npy",allow_pickle=True)
x_pred = np.load('../data/LPD_competition/npy/test2.npy',allow_pickle=True)
y = np.load("../data/LPD_competition/npy/Lotte_train_y5.npy",allow_pickle=True)
# y1 = np.zeros((len(y), len(y.unique())))
# for i, digit in enumerate(y):
#     y1[i, digit] = 1


x = preprocess_input(x) # (48000, 255, 255, 3)
x_pred = preprocess_input(x_pred)   # 



idg = ImageDataGenerator(
    # rotation_range=10, acc 하락
    width_shift_range=(-1,1),   # 0.1 => acc 하락
    height_shift_range=(-1,1),  # 0.1 => acc 하락
    # rotation_range=40, acc 하락 
    shear_range=0.2)    # 현상유지
    # zoom_range=0.2, acc 하락
    # horizontal_flip=True)

idg2 = ImageDataGenerator()

'''
- rotation_range: 이미지 회전 범위 (degrees)
- width_shift, height_shift: 그림을 수평 또는 수직으로 랜덤하게 평행 이동시키는 범위 
                                (원본 가로, 세로 길이에 대한 비율 값)
- rescale: 원본 영상은 0-255의 RGB 계수로 구성되는데, 이 같은 입력값은 
            모델을 효과적으로 학습시키기에 너무 높습니다 (통상적인 learning rate를 사용할 경우). 
            그래서 이를 1/255로 스케일링하여 0-1 범위로 변환시켜줍니다. 
            이는 다른 전처리 과정에 앞서 가장 먼저 적용됩니다.
- shear_range: 임의 전단 변환 (shearing transformation) 범위
- zoom_range: 임의 확대/축소 범위
- horizontal_flip`: True로 설정할 경우, 50% 확률로 이미지를 수평으로 뒤집습니다. 
    원본 이미지에 수평 비대칭성이 없을 때 효과적입니다. 즉, 뒤집어도 자연스러울 때 사용하면 좋습니다.
- fill_mode 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
'''

y = np.argmax(y, axis=1)

skf = StratifiedKFold(n_splits=2, random_state=42, shuffle=True)

val_loss_min = []
result = 0
nth = 0

for train_index, valid_index in skf.split(x,y) :
    
    mc = ModelCheckpoint('C:/data/LPD_competition/modelcheckpoint/lotte_0326.h5',save_best_only=True, verbose=1)
    
    x_train = x[train_index]
    x_valid = x[valid_index]    
    y_train = y[train_index]
    y_valid = y[valid_index]
    

    train_generator = idg.flow(x_train,y_train,batch_size=16, seed = 2048)
    # seed => random_state
    valid_generator = idg2.flow(x_valid,y_valid)
    test_generator = x_pred


    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation
    efficientnetb7 = EfficientNetB7(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])
    efficientnetb7.trainable = False
    a = efficientnetb7.output
    a = GlobalAveragePooling2D() (a)
    a = Flatten() (a)
    a = Dense(128) (a)
    a = BatchNormalization() (a)
    a = Activation('relu') (a)
    a = Dense(64) (a)
    a = BatchNormalization() (a)
    a = Activation('relu') (a)
    a = Dense(1000, activation= 'softmax') (a)

    model = Model(inputs = efficientnetb7.input, outputs = a)

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    early_stopping = EarlyStopping(patience= 15)
    lr = ReduceLROnPlateau(patience= 7, factor=0.5)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=1e-5,epsilon=None),
                    metrics=['acc'])
    learning_history = model.fit_generator(train_generator,epochs=1, 
        validation_data=valid_generator, callbacks=[early_stopping,lr,mc])
    
    # predict
    model.load_weights('C:/data/LPD_competition/modelcheckpoint/lotte_0326.h5')
    result += model.predict_generator(test_generator,verbose=True)/2
    
    # save val_loss
    # hist = pd.DataFrame(learning_history.history)
    # val_loss_min.append(hist['val_loss'].min())
    
    nth += 1
    print(nth, '번째 학습을 완료했습니다.')
print(result.shape)
sub = pd.read_csv('C:/data/LPD_competition/sample.csv')
sub['prediction'] = result
sub.to_csv('C:/data/LPD_competition/answer.csv',index=False)