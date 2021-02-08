import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split

dataset = load_iris()
x = dataset.data
y = dataset.target

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y = enc.fit_transform(y.reshape(-1,1)).toarray()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

#1. data and preprocessing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. model and compile
def build_model(drop=0.5, optimizer='adam', node=128):
    inputs = Input(shape=(4,), name='input')
    x = Dense(node, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(3, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    node = [128, 256, 512]
    return {"batch_size" : batches, "optimizer" : optimizers,
            "drop" : dropout, "node" : node}
hyperparameters = create_hyperparameters()
model2 = build_model()

# allows Grid/RandomizedSearch to understand keras models
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1) # epochs=3, validation_split=0.2) # also works here as well

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv=5)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=5)
re = ReduceLROnPlateau(monitor='val_loss', patience=3)
cp = ModelCheckpoint(filepath='C:/data/modelCheckpoint/k61_{epoch:02d}_{val_loss:.4f}.hdf5',
                     monitor='val_loss', save_best_only=True, mode='auto')

search.fit(x_train, y_train, verbose=1, epochs=1000, validation_split=0.2, callbacks=[es, re, cp]) # takes precedence over KerasClassifier

score = search.score(x_test, y_test)
print("Final score: ", score)
print(search.best_params_) # from the params that I choose
print(search.best_estimator_) # from all the params available
print(search.best_score_)

# Final score:  0.9666666388511658
# {'optimizer': 'adam', 'node': 128, 'drop': 0.2, 'batch_size': 20}
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001A17242DC40>
# 0.9416666626930237