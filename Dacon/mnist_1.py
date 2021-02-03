# 차근차근 하나하나 알아보도록 하자
# 어떻게 구성할거니? 
# CNN 모델을 돌릴 건데, ML이랑 연계해서 돌려보지 않으련?
# wrapper이라고 있는 거 같던데 좀 알아보고
# 내가 처음에 본 16번 벤치마킹하면서 다시 짜보자 내가 아는 건 두고 조금 된다싶으면 넣고 아니면 빼고 디버깅으로 어떻게든 해보자


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 2D Conv model
from tensorflow.keras.layers import Input, Convolution2D, Dense,Reshape, BatchNormalization, SpatialDropout2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model

#1. data
# load data from csv
filepath = 'C:/data/mnist/'
train = pd.read_csv(filepath + 'train.csv') # (2048, 787)
test  = pd.read_csv(filepath + 'test.csv') # (20480, 786)
submission = pd.read_csv(filepath + 'submission.csv') # (20480, 2)

# slicing
train_copy = train.copy()
x_train_pd = train_copy.loc[:, 'letter':'783'] 
y_train_pd = train_copy['digit'] 
x = x_train_pd.to_numpy() # (2048, 785)
y = y_train_pd.to_numpy() # (2048,)

test_copy = test.copy() # (20480, 786)
x_test_pd = test_copy.loc[:, 'letter':'783']
x_test = x_test_pd.to_numpy() # (20480, 785)

# convert string to float using LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 0] = le.fit_transform(x[:, 0]) # x.shape (2048, 785)
x_test[:, 0] = le.fit_transform(x_test[:, 0]) # (20480, 785)


# train_test_split
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
seed = 66
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=seed)

from xgboost import XGBClassifier

# dictionary within list
# parameters = [
#     {"n_estimators" : [100, 200, 300], "learning_rate" : [0.001, 0.01, 0.1, 0.3], "max_depth" : [4, 5, 6]},
#     {"n_estimators" : [90, 100, 110], "learning_rate" : [0.001, 0.01, 0.1], "max_depth" : [4, 5, 6], "colsample_bytree" : [0.6, 0.9, 1]},
#     {"n_estimators" : [90, 110], "learning_rate" : [0.001, 0.1, 0.5], "max_depth": [4,5,6], "colsample_bytree" : [0.6, 0.9, 1], "colsample_bylevel" : [0.6, 0.7, 0.9]}
# ]

#3. fit
# model = RandomizedSearchCV(XGBClassifier(n_jobs=-1, use_label_encoder=False), params, cv=5) # n_estimators
# model.fit(x_train, y_train, verbose=1, eval_metric=['mlogloss', 'merror'], eval_set=[(x_train, y_train), (x_val, y_val)], early_stopping_rounds=5)
# print('최적의 매개변수: {0}'.format(model.best_estimator_))

# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
# model = KerasRegressor(Keras_Conv2D, epochs=150, batch_size=32, verbose=1, shuffle=True)
# https://docs.python.org/3/tutorial/controlflow.html#keyword-arguments
# **argument --> dict 자료형으로 argument가 feed됨

# train_preds = cross_val_predict(model, x_train, y_train, cv=5)
# print(f"Score : {accuracy_score(y_train, np.argmax(train_preds, axis=1))}")

dropout_rate=0.5

model_in = Input(shape=x_train.shape[1])

image_in = model_in[:,:-26]
image_in = Reshape((28,28,1))(image_in)

x = Convolution2D(64, 3, padding='same')(image_in)
x_res = x
x = BatchNormalization()(x)
x = SpatialDropout2D(dropout_rate)(x)
x = MaxPooling2D()(x)
x = Flatten()
model_out = Dense(24, activation='softmax')(x)
model = Model(model_in, model_out)

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

from tensorflow.keras.optimizers import Adam
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=1024, callbacks=[early_stopping], validation_split=0.2)

#4. score


from sklearn.metrics import accuracy_score
y_pred = model.predict(x_val)
print('최종정답률: {0}'.format(accuracy_score(y_val, y_pred)))

# Submission 컬럼에 이를 기록
submission.digit = model.predict(x_test)

# 파일로 저장 후 업로드
submission.to_csv('my_submission.csv', index=False)

import joblib
joblib.dump(model, './mnist.joblib.dat')
print('joblib 저장하기 완료')

'''
# 57.84313725% 의 결과를 얻음
'''

# 성능 노답
# 최종정답률: 0.5097560975609756

# ImageDatagenerator & data augmentation
# idg = ImageDataGenerator(height_shift_range=(-1,1),width_shift_range=(-1,1)) # 이미지 카테고리화(4차원만 가능)
# idg2 = ImageDataGenerator() #ImageDataGenerator 머신러닝
# width_shift_range 좌우로 움직이는 정도:(-1,1) 처음부터 끝까지
# height_shift_range 위아래로 움직이는 정도

'''
# 16위
https://dacon.io/competitions/official/235626/codeshare/1668
# Final 16th/Public:0.921569/Stacking Random Model
'''