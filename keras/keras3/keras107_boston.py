# regressor

import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,\
    ReduceLROnPlateau
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#1. data
dataset = load_boston()
x = dataset.data # (506, 13)
y = dataset.target # (506, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
_, x_val, _, y_val = train_test_split(x_train, y_train, train_size=0.8)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#2. modeling
model = ak.StructuredDataRegressor(
    overwrite=True,
    max_trials=2
)

#3. fit
es = EarlyStopping(monitor='val_loss', mode='min', patience=6)
lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=2)
# ck = ModelCheckpoint('./temp/', save_best_only=True,
#     save_weights_only=True, monitor='val_loss', verbose=1
# )

model.fit(x_train, y_train,
    epochs=500,
    validation_data=(x_val, y_val),
    callbacks=[es, lr]
)

#4. evaluate
results = model.evaluate(x_test, y_test)
print(results)

#5. save model
model2 = model.export_model()
try:
    model2.save("./keras/keras3/save/model2_boston/", save_format="tf")
except Exception:
    model2.save("model_autokeras.h5")

best_model = model.tuner.get_best_model()
best_model.save("./keras/keras3/save/bestmodel_boston/", save_format="tf")