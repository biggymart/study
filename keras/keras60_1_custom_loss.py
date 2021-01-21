import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def custom_mse(y_true, y_pred): ### Point 1 ###
    return tf.math.reduce_mean(tf.square(y_true - y_pred))
# tensor 1.0 버전에서는 이런 식으로 사용, 표준편차

#1. data
x = np.array([1,2,3,4,5,6,7,8]).astype('float32') # (8,)
y = np.array([1,2,3,4,5,6,7,8]).astype('float32') # (8,)

#2. model
model = Sequential()
model.add(Dense(10, input_shape=(1,)))
model.add(Dense(10))
model.add(Dense(1))

#3. compile and fit
model.compile(loss=custom_mse, optimizer='adam')
model.fit(x, y, batch_size=1, epochs=30)

loss = model.evaluate(x, y)
print(loss)

# custom_mse
# loss='mse'의 비밀을 알아냈다
# 인자는 어디서 받느냐? 훈련을 하면서 자동으로 넣어준다
# 순서대로 1인자는 원래값, 2인자는 예측값 (이름 상관없고 순서만 중요)
# 0.03415842354297638

# 이것으로 인해 quantile을 풀 수 있나?
# 3개 인자를 넣으려고 lambda를 쓴다