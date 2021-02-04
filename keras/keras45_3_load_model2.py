# keras51_2_load_model.py 카피

#1. data
import numpy as np
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#1-1. preprocessing
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255. 
x_test = x_test.reshape(10000, 28, 28, 1)/255.

from tensorflow.keras.utils import to_categorical 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. model
#3. compile and fit
from tensorflow.keras.models import load_model
model = load_model('../data/h5/k51_1_model2.h5') # Point2 모델을 불러온다
# 결과가 epoch 돌아가는 것 없이 바로 나오는 것을 확인할 수 있다, 즉 모델과 가중치가 저장된 것임
# Point1 모델을 불러오는 경우 에러 발생, compile and fit을 하지 않았기 때문

# evaluate and predict
loss = model.evaluate(x_test, y_test)
print("[categorical_crossentropy, acc] :", loss)

y_pred = model.predict(x_test)
idx = 10
for i in range(idx):
    print(np.argmax(y_test[i]), np.argmax(y_pred[i]), end='/')

# 결과
# [categorical_crossentropy, acc] : [0.07746073603630066, 0.9764999747276306]
# 7 7/2 2/1 1/0 0/4 4/1 1/4 4/9 9/5 6/9 9/