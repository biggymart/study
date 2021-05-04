# tutorial url: https://autokeras.com/tutorial/overview/

import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

model = ak.ImageClassifier(
    overwrite=True, # Input shape도 없고, OneHotEncoding 안 해도 돌아감
    max_trials=2  # 튜닝할 게 있다면 유일하게 이 부분
)

model.fit(x_train, y_train, epochs=3)

results = model.evaluate(x_test, y_test)

print(results)