# 10개의 판별 모델을 만들어서 competition 벌이는 대환장 소스코드!

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from natsort import natsorted
import pandas as pd
import cv2 as cv
import numpy as np
import os
import gc

##### 변수 및 하이퍼파라미터 조절기 #####
TRAIN_DIR = 'C:/data/LPD_competition/train'
TEST_DIR = 'C:/data/LPD_competition/test'
model_path = 'C:/data/LPD_competition/modelcheckpoint/lotte_0322_1_{epoch:02d}-{val_loss:.4f}.hdf5'

DIMENSION = 128
atom = 100 # 각 모델이 구분할 class의 개수

train_fnames = natsorted(os.listdir(TRAIN_DIR))
test_fnames = natsorted(os.listdir(TEST_DIR))

NODE = 4096
DROPOUT_RATE = 0.2
#######################################

##### 제너레이터 정의 (Initialize image data generator) #####
########################
train_datagen = ImageDataGenerator(
    validation_split = 0.2,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    horizontal_flip=True,
    rotation_range=5,
    zoom_range=0.2,
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
########################


# 각 폴더에 있는 사진을 뻥튀기해줌
def augment_img(multiply_by=2): # 아무 값 안 주면 2배로 뿔려줌
    for idx, folder in enumerate(train_fnames): # 트레인 폴더는 1000개 있다
        if idx >= 1: # 폴더 X개만 해볼까
            break
        # print("folder :", folder)
        base_dir = TRAIN_DIR + '/' + folder + '/' # 각 폴더 디렉토리 'C:/data/LPD_competition/train/0'
        # print("basedir :", base_dir)
        img_lst = natsorted(os.listdir(base_dir)) # 각 폴더 안에 있는 jpg 파일 리스트
        # print(img_lst)

        for i, f in enumerate(img_lst): # 각 폴더에 접근해서
            # if i >= 1:
            #     break
            # print("f :", f)
            img_dir = base_dir + f # 'C:/data/LPD_competition/train/0/0.jpg'
            # print("img_dir :", img_dir)
            img = np.expand_dims(image.load_img(img_dir, target_size=(DIMENSION, DIMENSION)), axis=0) # 각 이미지를 불어온다
            # print(img, "\n", img.shape)
            # cv_img = cv.imread(img_dir)
            # cv.imshow("whatever", cv_img)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            train_datagen.fit(img)
            for x, val in zip(train_datagen.flow(x=img,
                save_to_dir=base_dir, # this is where we figure out where to save
                save_prefix='aug', # it will save the images as 'aug_0912' some number for every new augmented image
                shuffle=False), range(multiply_by)) : # here we define a range because we want 10 augmented images otherwise it will keep looping forever I think
                pass # 출처: https://stackoverflow.com/questions/47826730/how-to-save-resized-images-using-imagedatagenerator-and-flow-from-directory-in-k
            print("base dir :", base_dir, "finished ", x)


# 1000의 약수인지 확인해주는 함수
def test_atom(atom):
    if len(train_fnames) % atom == 0:
        print("atom well set!")
    else: 
        raise ValueError('atom should evenly divide 1000')

# 10개 모델을 훈련할 데이터 각 100 클래스로 구성해서 npy로 저장하는 함수
def mk_npy(atom):
    for i in range(int(len(train_fnames)/atom)):
        xy_train = train_datagen.flow_from_directory(
            directory=TRAIN_DIR,
            class_mode='categorical',
            classes=train_fnames[i:i+atom],
            subset='training',
            target_size=(DIMENSION, DIMENSION)
        )
        xy_val = train_datagen.flow_from_directory(
            directory=TRAIN_DIR,
            class_mode='categorical',
            classes=train_fnames[i:i+atom],
            subset='validation',
            target_size=(DIMENSION, DIMENSION)
        )
        np.save('../data/LPD_competition/npy/Lotte_train_x_{0}.npy'.format(i+1), arr=xy_train[0][0])
        np.save('../data/LPD_competition/npy/Lotte_train_y_{0}.npy'.format(i+1), arr=xy_train[0][1])
        np.save('../data/LPD_competition/npy/Lotte_val_x_{0}.npy'.format(i+1), arr=xy_val[0][0])
        np.save('../data/LPD_competition/npy/Lotte_val_y_{0}.npy'.format(i+1), arr=xy_val[0][1])
        print('npy files saved for ', i+1, 'th time')

# npy 불러와서 모델 컴파일, 훈련, h5 세이브하는 함수
def mk_h5():
    for i in range(int(len(train_fnames)/atom)):
        x_train = np.load('../data/LPD_competition/npy/Lotte_train_x_{0}.npy'.format(i+1))
        y_train = np.load('../data/LPD_competition/npy/Lotte_train_y_{0}.npy'.format(i+1))
        x_val = np.load('../data/LPD_competition/npy/Lotte_val_x_{0}.npy'.format(i+1))
        y_val = np.load('../data/LPD_competition/npy/Lotte_val_y_{0}.npy'.format(i+1))
        print('npy loaded for', i+1, 'th time')

        initial_model = MobileNetV2(weights="imagenet", include_top=False, input_shape = (DIMENSION, DIMENSION, 3))
        last = initial_model.output

        x = Flatten()(last)
        x = Dense(NODE, activation='relu')(x)
        x = Dropout(DROPOUT_RATE)(x)
        x = Dense(NODE, activation='relu')(x)
        preds = Dense(atom, activation='softmax')(x)
        model = Model(initial_model.input, preds)

        # compile
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # fit
        es = EarlyStopping(monitor='val_loss', patience=10)
        re = ReduceLROnPlateau(monitor='val_loss', patience=5)
        cp = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, mode='auto')

        model.fit(x_train, y_train, epochs=30, validation_data=(x_val, y_val), callbacks=[es, re, cp])
        model.save('../data/LPD_competition/h5/Lotte_model_{0}.h5'.format(i+1))
        print('model saved for', i+1, 'th time')
        gc.collect()


#####################################작업해야 할 부분######################################################################

# 한 이미지에 대해서 10개 모델로 평가해주고 리스트로 각 값 반환
def predict(image_path):
    h5_dir = 'C:/data/LPD_competition/h5'
    os.listdir(h5_dir)
    
    # 10개의 값이 들어간 리스트가 되어야
    pred_lst = []
    
    for i, f in enumerate(h5_dir):
        if i >= 1:
            break

        # 모델에 데이터 shape 맞도록 전처리
        test_img = image.load_img(image_path, target_size=(DIMENSION, DIMENSION))
        x = image.img_to_array(test_img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # 모델 하나씩 불러옴
        model = load_model('../data/LPD_competition/h5/Lotte_model_{0}.h5'.format(i+1))
        # loss = model.evaluate(x_test, y_test)
        # print("[categorical_crossentropy, acc] :", loss)
        y_pred = model.predict(x)
        print("y_pred의 shape :", y_pred.shape)
        print(i, "번째 모델의 인덱스는", np.argmax(y_pred),\
             "\n그 예측 정확도는 ", y_pred[0, np.argmax(y_pred)])
        print("append한 값 :", (i * atom) + np.argmax(y_pred))
        pred_lst.append((i * atom) + np.argmax(y_pred))
        gc.collect()
    return pred_lst

##### 함수 호출 (커맨드센터) #####
test_atom(atom)
augment_img(multiply_by=5)
mk_npy(atom)
mk_h5()

# predict함수는 각 이미지에 대해서 처리해주는 거니까
# 72000개 한번에 하지 말고 100개씩 해볼까 싶네
for i, f in enumerate(test_fnames):
    # 파일 X개만 시도해볼까
    if i >= 1:
        break

    prediction = predict(image_path=os.path.join(TEST_DIR, f))
    print(prediction)

# sub = pd.read_csv('C:/data/LPD_competition/sample.csv')
# # sub['prediction'] = np.argmax(prediction, axis = 1)
# sub['prediction'] = max(prediction, axis = 1)
# sub.to_csv('C:/data/csv/lotte0317_3.csv',index=False)



