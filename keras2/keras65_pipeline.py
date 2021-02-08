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

from tensorflow.keras.models import save_model
save_model(search, filepath='../data/h5/k64.h5')


# Final score:  0.9860000014305115
# {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 40}
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000026F0D121E20>
# 0.9804500142733256
# {'mean_fit_time': array([ 80.78310458,  16.1492552 , 217.34447304, 278.15846189,
#        218.27508656,  28.53572003,  19.36032772,  21.41552202,
#         33.24328272,  23.03036928]), 'std_fit_time': array([4.97633911, 1.17580648, 0.99905162, 2.26659108, 1.00987607,
#        1.97611169, 1.39250416, 2.90685582, 0.90435215, 1.30821381]), 'mean_score_time': array([2.35113835, 0.69550284, 1.06699347, 1.22564356, 1.06532892,
#        1.09105794, 0.68022052, 1.28213072, 1.06567844, 1.24383648]), 'std_score_time': array([0.01710891, 0.00394651, 0.00665468, 0.0040017 , 0.01217912,
#        0.06232058, 0.00772865, 0.05342676, 0.01181792, 0.01156329]), 'param_optimizer': masked_array(data=['adam', 'adam', 'adadelta', 'adadelta', 'adadelta',
#                    'adam', 'adam', 'rmsprop', 'adam', 'rmsprop'],
#              mask=[False, False, False, False, False, False, False, False,
#                    False, False],
#        fill_value='?',
#             dtype=object), 'param_drop': masked_array(data=[0.3, 0.1, 0.2, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.3],
#              mask=[False, False, False, False, False, False, False, False,
#                    False, False],
#        fill_value='?',
#             dtype=object), 'param_batch_size': masked_array(data=[10, 40, 30, 20, 30, 30, 40, 20, 30, 20],
#              mask=[False, False, False, False, False, False, False, False,
#                    False, False],
#        fill_value='?',
#             dtype=object), 'params': [{'optimizer': 'adam', 'drop': 0.3, 'batch_size': 10}, {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 40}, {'optimizer': 'adadelta', 'drop': 0.2, 'batch_size': 30}, {'optimizer': 'adadelta', 'drop': 0.2, 'batch_size': 20}, {'optimizer': 'adadelta', 'drop': 0.3, 'batch_size': 30}, {'optimizer': 'adam', 'drop': 0.2, 'batch_size': 30}, {'optimizer': 'adam', 'drop': 0.3, 'batch_size': 40}, {'optimizer': 'rmsprop', 'drop': 0.2, 'batch_size': 20}, {'optimizer': 'adam', 'drop': 0.3, 'batch_size': 30}, {'optimizer': 'rmsprop', 'drop': 0.3, 'batch_size': 20}], 'split0_test_score': array([0.98150003, 0.98185003, 0.9149    , 0.92079997, 0.90555   ,
#        0.9817    , 0.98220003, 0.97579998, 0.98154998, 0.97535002]), 'split1_test_score': array([0.97944999, 0.98000002, 0.90925002, 0.91415   , 0.89929998,
#        0.9799    , 0.97955   , 0.97549999, 0.98035002, 0.97280002]), 'split2_test_score': array([0.97939998, 0.9795    , 0.91734999, 0.92025   , 0.90665001,
#        0.97895002, 0.97764999, 0.9738    , 0.97850001, 0.97215003]), 'mean_test_score': array([0.98011667, 0.98045001, 0.91383334, 0.91839999, 0.90383333,
#        0.98018334, 0.97980001, 0.97503332, 0.98013333, 0.97343336]), 'std_test_score': array([0.0009784 , 0.00101079, 0.00339173, 0.00301357, 0.00323687,
#        0.00114041, 0.00186594, 0.00088065, 0.00125454, 0.00138102]), 'rank_test_score': array([ 4,  1,  9,  8, 10,  2,  5,  6,  3,  7])}

# Traceback (most recent call last):
#   File "c:\Study\keras2\keras64_save.py", line 66, in <module>
#     search.save('../data/h5/k64.h5')
# AttributeError: 'RandomizedSearchCV' object has no attribute 'save'