# keras60_2_quantile_loss.py 카피

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 텐서플로우 딥러닝 Quantile Regression 참고
def quantile_loss_dacon(q, y_true, y_pred):
	err = (y_true - y_pred)
	return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

#1. data
x = np.array([1,2,3,4,5,6,7,8]).astype('float32') # (8,)
y = np.array([1,2,3,4,5,6,7,8]).astype('float32') # (8,)

#2. model
model = Sequential()
model.add(Dense(10, input_shape=(1,)))
model.add(Dense(10))
model.add(Dense(1))

#3. compile and fit
model.compile(loss = lambda y_true, y_pred: quantile_loss_dacon(qs[0], y_true, y_pred), optimizer='adam') ### Point2 ###
model.fit(x, y, batch_size=1, epochs=30)

loss = model.evaluate(x, y)
print(loss)

# 결과 quantile 0.1의 결과임
# 0.0073398323729634285

