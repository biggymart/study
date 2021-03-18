# 63 파이프라인으로 구성
# .255 로 나누는 것 말고... 오키?
# 함정이 있다고 한다
# 그리드서치도 사용할 것

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold

#1. data and preprocessing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

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
    batches = [30, 50]
    optimizers = ['rmsprop', 'adam']
    dropout = [0.2, 0.3]
    return {"clf__batch_size" : batches, 
            "clf__optimizer" : optimizers,
            "clf__drop" : dropout}
hyperparameters = create_hyperparameters()

# allows Grid/RandomizedSearch to understand keras models
model2 = KerasClassifier(build_fn=build_model, verbose=1) 

pipe = Pipeline([("scaler", MinMaxScaler()), ('clf', model2)])

kfold = KFold(n_splits=3, shuffle=True, random_state=66)
search = RandomizedSearchCV(pipe, hyperparameters, cv=kfold)

es = EarlyStopping(monitor='val_loss', patience=5)
re = ReduceLROnPlateau(monitor='val_loss', patience=3)
cp = ModelCheckpoint(filepath='C:/data/modelCheckpoint/k65_{epoch:02d}_{val_loss:.4f}.hdf5',
           monitor='val_loss', save_best_only=True, mode='auto')

search.fit(x_train, y_train, clf__epochs=30, 
           clf__validation_split=0.2, clf__callbacks=[es, re, cp])


print("best_params : ", search.best_params_)         
print("best_estimator : ", search.best_estimator_)   
print("best_score : ", search.best_score_)           

acc = search.score(x_test, y_test)
print("Score : ", acc)


# best_params :  {'clf__optimizer': 'adam', 'clf__drop': 0.2, 'clf__batch_size': 30}
# best_estimator :  Pipeline(steps=[('scaler', MinMaxScaler()),
#                 ('clf', <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000021AE8F4CBB0>)])
# best_score :  0.9816333254178365
# Score :  0.9840999841690063