# keras60_1_custom_loss.py 카피

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def custom_mse(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))

def quantile_loss(y_true, y_pred): ### Point 1 ###
    qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    q = tf.constant(np.array([qs]), dtype=tf.float32) # 리스트의 텐서 형식의 상수화
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e) # Pinball loss
    return K.mean(v) # 하나만 반환

#1. data
x = np.array([1,2,3,4,5,6,7,8]).astype('float32') # (8,)
y = np.array([1,2,3,4,5,6,7,8]).astype('float32') # (8,)

#2. model
model = Sequential()
model.add(Dense(10, input_shape=(1,)))
model.add(Dense(10))
model.add(Dense(1))

#3. compile and fit
model.compile(loss=custom_mse, optimizer='adam') ### Point2 ###
model.fit(x, y, batch_size=1, epochs=30)

loss = model.evaluate(x, y)
print(loss)

# custom_mse
# 0.03415842354297638

# quantile_loss
# 0.00638581020757556
# loss 지표 자체가 바꼈기 때문에 더 좋아졌는지 판단하기 어려움
# 모델을 9번을 돌려서 submit에 넣으면 됨
# 0.5는 평균임