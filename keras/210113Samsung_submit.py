#1. data
import numpy as np
x_test = np.load('../data/npy/samsung_x_test.npy')
y_test = np.load('../data/npy/samsung_y_test.npy')

#2. model
#3. compile and fit
from tensorflow.keras.models import load_model
model = load_model('../data/h5/samsung_lstm.h5')

#4. evaluate and predict
loss = model.evaluate(x_test, y_test, batch_size=32)
print("loss(mse, mae) :", loss)

x_pred = np.load('../data/npy/samsung_x_pred.npy')

y_pred = model.predict(x_pred, batch_size=32)
print(y_pred)