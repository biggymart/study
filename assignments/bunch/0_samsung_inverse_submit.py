#1. data
import numpy as np
npz_loaded1 = np.load('../data/npy/samsung_inverse1.npz')
x1_test = npz_loaded1['x1_test']
y1_test = npz_loaded1['y1_test']
x1_pred = npz_loaded1['x1_pred']
npz_loaded2 = np.load('../data/npy/samsung_inverse2.npz')
x2_test = npz_loaded2['x2_test']
x2_pred = npz_loaded2['x2_pred']

#2. model
#3. compile and fit
from tensorflow.keras.models import load_model
model = load_model('../data/h5/samsung_inverse.h5')

#4. evaluate and predict
loss = model.evaluate([x1_test, x2_test], y1_test, batch_size=32)
y_pred = model.predict([x1_pred, x2_pred], batch_size=32)
print(loss, y_pred)

# 결과
# [3929277.75, 1491.656005859375] [[91563.38] [90250.57]]