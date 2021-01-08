# 2:3 앙상블 (ensemble)
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate # Concatenate, LSTM, Conv2D


#1. data
x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
y1 = np.array([range(711,811), range(1,101), range(201, 301)])
y2 = np.array([range(501, 601), range(711, 811), range(100)])
y3 = np.array([range(601, 701), range(811, 911), range(1100, 1200)])

x1 = np.transpose(x1); x2 = np.transpose(x2)
y1 = np.transpose(y1); y2 = np.transpose(y2); y3 = np.transpose(y3)

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(
    x1, x2, y1, y2, y3, train_size=0.8, shuffle=False)

#2. model
# 입력 모델
input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(5, activation='relu')(dense1)

input2 = Input(shape=(3,))
dense2 = Dense(10, activation='relu')(input2)
dense2 = Dense(5, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)

# 모델 병합
merge1 = concatenate([dense1, dense2])
middle1 = Dense(30)(merge1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)

# 출력 모델
output1 = Dense(30, name="out1_layer1")(middle1)
output1 = Dense(7, name="out1_layer2")(output1)
output1 = Dense(3, name="out1_layer3")(output1)

output2 = Dense(15, name="out2_layer1")(middle1)
output2 = Dense(7, name="out2_layer2")(output2)
output2 = Dense(7, name="out2_layer3")(output2)
output2 = Dense(3, name="out2_layer4")(output2)

output3 = Dense(10, name="out3_layer1")(middle1)
output3 = Dense(3, name="out3_layer2")(output3)

# 모델 선언
model = Model(inputs=[input1, input2], outputs=[output1, output2, output3])
# model.summary()

#3. compile and fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], [y1_train, y2_train, y3_train], epochs=100, batch_size=1, validation_split=0.2, verbose=0)

#4. evaluate and predict
loss = model.evaluate([x1_test, x2_test], [y1_test, y2_test, y3_test], batch_size=1)
print(model.metrics_names, '\n', loss)

y1_predict, y2_predict, y3_predict = model.predict([x1_test, x2_test])
# print("y1_predict :\n", y3_predict) # shape (20,3)의 결과가 출력됨

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', (RMSE(y1_test, y1_predict) + RMSE(y2_test, y2_predict) + RMSE(y3_test, y3_predict))/int(len(loss)/2))

# R2 구하기
from sklearn.metrics import r2_score
print('R2 :', (r2_score(y1_test, y1_predict) + r2_score(y2_test, y2_predict) + r2_score(y3_test, y3_predict))/int(len(loss)/2))

# # 결과
# ['loss', 'dense_10_loss', 'dense_14_loss', 'dense_16_loss', 'dense_10_mae', 'dense_14_mae', 'dense_16_mae'] 
# [0.06699274480342865, 0.007552322931587696, 0.026195675134658813, 0.03324474021792412, 0.04107614979147911, 0.07659034430980682, 0.09259440004825592]
# RMSE : 0.14369785232636179
# R2 : 0.9993283537878018