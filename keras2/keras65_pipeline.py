# 63 파이프라인으로 구성
# .255 로 나누는 것 말고... 오키?
# 함정이 있다고 한다
# 그리드서치도 사용할 것

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#1. data and preprocessing
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from sklearn.model_selection import KFold
kfold = KFold(n_splits=3, shuffle=True, random_state=66)

x_train = x_train.reshape(60000, 28*28).astype('float32')
x_test = x_test.reshape(10000, 28*28).astype('float32')

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
    return {"aaa__batch_size" : batches, "aaa__optimizer" : optimizers,
            "aaa__drop" : dropout}
hyperparameters = create_hyperparameters()
model2 = build_model()

# allows Grid/RandomizedSearch to understand keras models
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1) 

from sklearn.pipeline import Pipeline, make_pipeline
pipe = Pipeline([("scaler", scaler), ('aaa', model2)])
pipe.get_params().keys()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(pipe, hyperparameters, cv=kfold)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=5)
re = ReduceLROnPlateau(monitor='val_loss', patience=3)
cp = ModelCheckpoint(filepath='C:/data/modelCheckpoint/k61_{epoch:02d}_{val_loss:.4f}.hdf5', monitor='val_loss', save_best_only=True, mode='auto')

search.fit(x_train, y_train, aaa__epochs=100, aaa__validation_split=0.2, aaa__callbacks=[es, re, cp])

acc = search.score(x_test, y_test)
print("Final score: ", acc)
print(search.best_params_)
print(search.best_estimator_)
print(search.best_score_)
print(search.cv_results_)

# save model
import pickle
pickle.dump(search, open('../data/h5/k65.pickle.dat', 'wb')) #dump == save, write binary
print('pickle 저장 완료')
