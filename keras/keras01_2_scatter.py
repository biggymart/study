# scatter : 흩뿌리다

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np


#1. data
x = np.arange(1,11)
y = np.array([1,2,4,3,5,5,7,9,8,11])
print(x)
print(y)

#2. model
#2-2 condition 2 (with hidden layer)
model = Sequential()
model.add(Dense(1, input_shape=(1,)))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

'''
#2-1 condition 1 (without hidden layer)
model = Sequential()
model.add(Dense(1, input_shape=(1,)))
'''

#3. compile and fit
optimizer = RMSprop(learning_rate=0.01)

model.compile(loss='mse', optimizer=optimizer)
model.fit(x, y, epochs=1000) 
# 조건1> 히든레이어 없이
# epochs 값을 1, 10, 100, 200, 1000으로 바꿔가며 실습해보자
# 히든레이어가 없으니까 너무 안 맞네

# 조건2> 히든레이어 추가해서
# epochs 값을 10만 주어도 잘 찾아감

# lesson learned: machine learning does not have hidden layer
# thus, calculation speed is fast, requiring less burden

y_pred = model.predict(x)

plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.show()