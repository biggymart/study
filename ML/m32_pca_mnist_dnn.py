# Task requirement:
# Using n_components ((1) evr >= 0.95 (2) evr >= 1.0) found in m31 file, make a DNN model.
# Its result should be better than the model in keras40_mnist3_dnn.py.
# Compare with CNN model (keras40_mnist2_cnn.py).

from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical
import numpy as np

#1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28 * 28).astype('float32')/255.
x_test = x_test.reshape(10000, 28 * 28)/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

n_compo = 154 # 713 for evr >= 1.0
pca = PCA(n_components=n_compo)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.fit_transform(x_test)

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.activations import relu, softmax

model = Sequential()
model.add(Dense(32, input_shape=(n_compo,), activation=relu))
model.add(Dense(256, activation=relu))
model.add(Dropout(0.5))
model.add(Dense(10, activation=softmax))

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

from tensorflow.keras.optimizers import Adam
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['acc'])
model.fit(x_train_pca, y_train, epochs=100, batch_size=32, callbacks=[early_stopping], validation_split=0.2)

# evaluate and predict
loss = model.evaluate(x_test_pca, y_test)
print("[categorical_crossentropy, acc] :", loss)

y_pred = model.predict(x_test_pca)
idx = 10
for i in range(idx):
    print(np.argmax(y_test[i]), np.argmax(y_pred[i]), end='/')

# keras40_mnist3_dnn.py results
# [categorical_crossentropy, acc] : [0.1446416974067688, 0.9541000127792358]
# 7 7/2 2/1 1/0 0/4 4/1 1/4 4/9 9/5 6/9 9/

# m32_pca_mnist1_dnn.py
# evr >= 0.95
# [categorical_crossentropy, acc] : [6.0644145011901855, 0.13860000669956207]
# 7 5/2 9/1 7/0 0/4 2/1 7/4 3/9 2/5 6/9 3/

# evr >= 1.00
# [categorical_crossentropy, acc] : [15.62626838684082, 0.14300000667572021]
# 7 1/2 7/1 7/0 0/4 2/1 7/4 3/9 2/5 6/9 5/