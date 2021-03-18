# keras61 카피,
# model.cv_results를 붙여서 완성

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
model2 = KerasClassifier(build_fn=build_model, verbose=1) 

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv=3)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=5)
re = ReduceLROnPlateau(monitor='val_loss', patience=3)
cp = ModelCheckpoint(filepath='C:/data/modelCheckpoint/k61_{epoch:02d}_{val_loss:.4f}.hdf5', monitor='val_loss', save_best_only=True, mode='auto')

search.fit(x_train, y_train, verbose=1, epochs=100, validation_split=0.2, callbacks=[es, re, cp])

acc = search.score(x_test, y_test)
print("Final score: ", acc)
print(search.best_params_)
print(search.best_estimator_)
print(search.best_score_)
print(search.cv_results_)

# Final score:  0.9861000180244446
# {'optimizer': 'adam', 'drop': 0.2, 'batch_size': 50}
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000024B0A5BBFD0>
# 0.9804499944051107
# {'mean_fit_time': array([132.05653318,  11.71551363,  19.43325631, 109.18705487,
#         15.36306047, 109.62580315, 499.88461828,  11.86260557,
#        498.35198704, 212.69750706]), 'std_fit_time': array([0.64898272, 0.96789654, 0.09546966, 0.59897107, 1.27169695,
#        0.39227813, 1.41784274, 1.39523738, 4.9791177 , 0.42721433]), 'mean_score_time': array([0.69408758, 0.672273  , 1.09609143, 0.57389418, 0.58138696,
#        0.57467961, 2.37793692, 0.58324393, 2.34424909, 1.05056723]), 'std_score_time': array([0.01472241, 0.0042037 , 0.04895294, 0.01076015, 0.00497841,
#        0.00632072, 0.01135597, 0.00465782, 0.02062527, 0.00664391]), 'param_optimizer': masked_array(data=['adadelta', 'rmsprop', 'rmsprop', 'adadelta', 'adam',
#                    'adadelta', 'adadelta', 'rmsprop', 'adadelta',
#                    'adadelta'],
#              mask=[False, False, False, False, False, False, False, False,
#                    False, False],
#        fill_value='?',
#             dtype=object), 'param_drop': masked_array(data=[0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.2, 0.2, 0.3, 0.1],
#              mask=[False, False, False, False, False, False, False, False,
#                    False, False],
#        fill_value='?',
#             dtype=object), 'param_batch_size': masked_array(data=[40, 40, 30, 50, 50, 50, 10, 50, 10, 30],
#              mask=[False, False, False, False, False, False, False, False,
#                    False, False],
#        fill_value='?',
#             dtype=object), 'params': [{'optimizer': 'adadelta', 'drop': 0.1, 'batch_size': 40}, {'optimizer': 'rmsprop', 'drop': 0.1, 'batch_size': 40}, {'optimizer': 'rmsprop', 'drop': 0.2, 'batch_size': 30}, {'optimizer': 'adadelta', 'drop': 0.2, 'batch_size': 50}, {'optimizer': 'adam', 'drop': 0.2, 'batch_size': 50}, {'optimizer': 'adadelta', 'drop': 0.3, 'batch_size': 50}, {'optimizer': 'adadelta', 'drop': 0.2, 'batch_size': 10}, {'optimizer': 'rmsprop', 'drop': 0.2, 'batch_size': 50}, {'optimizer': 'adadelta', 'drop': 0.3, 'batch_size': 10}, {'optimizer': 'adadelta', 'drop': 0.1, 'batch_size': 30}], 'split0_test_score': array([0.92009997, 0.98000002, 0.97815001, 0.91224998, 0.98210001,
#        0.9016    , 0.92484999, 0.98035002, 0.91570002, 0.92374998]), 'split1_test_score': array([0.91759998, 0.97895002, 0.97820002, 0.90549999, 0.97944999,
#        0.89305001, 0.91939998, 0.97780001, 0.91035002, 0.9181    ]), 'split2_test_score': array([0.91874999, 0.97705001, 0.9763    , 0.91100001, 0.97979999,
#        0.90165001, 0.92594999, 0.97684997, 0.91775   , 0.92355001]), 'mean_test_score': array([0.91881665, 0.97866668, 0.97755001, 0.90958333, 0.98044999,
#        0.89876668, 0.92339998, 0.97833333, 0.91460001, 0.9218    ]), 'std_test_score': array([0.00102171, 0.00122089, 0.00088413, 0.0029321 , 0.00117545,
#        0.00404234, 0.00286386, 0.00147781, 0.00311955, 0.00261757]), 'rank_test_score': array([ 7,  2,  4,  9,  1, 10,  5,  3,  8,  6])}