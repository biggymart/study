# "Parameter search automization"
# Keras model ++ wrappers ++ Grid/RandomizedSearch (multiple parameters)

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#1. data and preprocessing
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

#2. model and compile
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28*28, ), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss='categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return {"batch_size" : batches, "optimizer" : optimizers,
            "drop" : dropout}
hyperparameters = create_hyperparameters()
model2 = build_model()

# allows Grid/RandomizedSearch to understand keras models
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1) # epochs=3, validation_split=0.2) # also works here as well

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv=3)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=5)
re = ReduceLROnPlateau(monitor='val_loss', patience=3)
cp = ModelCheckpoint(filepath='C:/data/modelCheckpoint/k61_{epoch:02d}_{val_loss:.4f}.hdf5', monitor='val_loss', save_best_only=True, mode='auto')

search.fit(x_train, y_train, verbose=1, epochs=100, validation_split=0.2, callbacks=[es, re, cp]) # takes precedence over KerasClassifier

acc = search.score(x_test, y_test)
print("Final score: ", acc)
print(search.best_params_) # from the params that I choose
print(search.best_estimator_) # from all the params available
print(search.best_score_)
print(search.cv_results_)

# save model
import pickle
pickle.dump(search, open('../data/h5/k64.pickle.dat', 'wb')) #dump == save, write binary
print('pickle 저장 완료')

# without wrappers
# TypeError: If no scoring is specified, the estimator passed should have
# a 'score' method. The estimator <tensorflow.python.keras.engine.functional.Functional object at 0x000001A9A9A0AFD0> does not.

# Final score:  0.9855999946594238
# {'optimizer': 'adam', 'drop': 0.3, 'batch_size': 50}
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x00000267821CB8E0>
# 0.9801833232243856

# you can also include node and activation as well by using variables in layers and adding them in parameters dict

# Tasks:
# make it into CNN model: (1) reshape data into 4D (2) layer Conv2D, MaxPool2D, Flatten
# make it into LSTM model: (1) reshape data into 3D (2) layer LSTM