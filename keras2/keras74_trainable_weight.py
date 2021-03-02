# 이제 모델의 가중치를 직접 볼 수 있다

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. data
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])

#2. model
model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

# print(model.weights)
# print(model.trainable_weights)

print(len(model.weights)) # 레이어당 kernel, bias (2개)가 있어서 4 * 2 = 8
print(len(model.trainable_weights))
