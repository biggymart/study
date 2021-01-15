#1. data
import numpy as np
# np_array=np.load('../data/npy/samsung.npz')
# print(np_array.files)

npz_loaded = np.load('../data/npy/samsung.npz')
x_test = npz_loaded['x_test']
y_test = npz_loaded['y_test']
x_pred = npz_loaded['x_pred']

#2. model
#3. compile and fit
from tensorflow.keras.models import load_model
model = load_model('../data/h5/samsung_lstm.h5')

#4. evaluate and predict
loss = model.evaluate(x_test, y_test, batch_size=32)
print("loss(mse, mae) :", loss)

y_pred = model.predict(x_pred, batch_size=32)
print(y_pred)