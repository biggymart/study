# tutorial url: https://autokeras.com/tutorial/overview/

import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,\
    ReduceLROnPlateau
from tensorflow.keras.models import load_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = load_model('./keras/keras3/save/aaa.h5')
model.summary()

best_model = load_model('./keras/keras3/save/best_aaa.h5')
best_model.summary()

# 위 두 모델은 동일한 것을 알 수 있다.
#########################################################

results = model.evaluate(x_test, y_test)
print(results)

best_results = best_model.evaluate(x_test, y_test)
print(best_results)



# model = ak.ImageClassifier(
#     overwrite=True, # Input shape도 없고, OneHotEncoding 안 해도 돌아감
#     max_trials=1,  # 튜닝할 게 있다면 유일하게 이 부분
#     loss='mse',
#     metrics=['acc']
# )

# model.summary()
# 에러 발생

# es = EarlyStopping(monitor='val_loss', mode='min', patience=6)
# lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=2)
# ck = ModelCheckpoint('./temp/', save_best_only=True,
#     save_weights_only=True, monitor='val_loss', verbose=1
# )

# model.fit(x_train, y_train, epochs=1, validation_split=0.2, 
#     callbacks=[es, lr, ck]
# )

# results = model.evaluate(x_test, y_test)

# print(results)

# # model.summary()
# # 에러 발생

# model2 = model.export_model()
# model2.save('./keras/keras3/save/aaa.h5')
# # AttributeError: 'ImageClassifier' object has no attribute 'save'
