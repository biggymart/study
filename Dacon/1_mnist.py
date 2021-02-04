import numpy as np
import pandas as pd
from numpy import expand_dims
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

########################################
# load data from csv
filepath = 'C:/data/mnist/'
train = pd.read_csv(filepath + 'train.csv') # (2048, 787)
test  = pd.read_csv(filepath + 'test.csv') # (20480, 786)
submission = pd.read_csv(filepath + 'submission.csv') # (20480, 2)
########################################

# slicing
x_train = train.loc[:, '0':'783'] 
y_train = train['digit'] 
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()

x_test = test.loc[:, '0':'783']
x_test = x_test.to_numpy()

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

#  ImageDataGenerator
idg1 = ImageDataGenerator(height_shift_range=(-1,1),width_shift_range=(-1,1))
idg2 = ImageDataGenerator()

sample_data = x_train[100].copy()
sample = expand_dims(sample_data,0)
sample_datagen = ImageDataGenerator(height_shift_range=(-1,1),width_shift_range=(-1,1))
sample_generator = sample_datagen.flow(sample, batch_size=1)

#2. Modeling
train_generator = idg1.flow(x_train, y_train, batch_size=16)
pred_generator = idg2.flow(x_test, shuffle=False)

#2. Modeling
model = Sequential()

model.add(Conv2D(16, (3,3), activation='relu', input_shape=(28, 28,1), padding='same'))
model.add(BatchNormalization()) 
model.add(Dropout(0.3))

model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization()) 
model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization()) 
model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization()) 
model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization()) 
model.add(MaxPooling2D(3,3))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization()) 
model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization()) 
model.add(MaxPooling2D(3,3))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

#3. Compile, Train
reLR = ReduceLROnPlateau(patience=5, verbose=1, factor=0.5, monitor='acc')
es = EarlyStopping(patience=10, verbose=1, monitor='acc')

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.01, epsilon=None), metrics=['acc'])
model.fit_generator(train_generator, epochs=1000, callbacks=[es, reLR])

#4. Evaluate, Predict
result = 0
result += model.predict(pred_generator, verbose=True)/40
print(result)
print(result.shape)

submission['digit'] = result.argmax(1)
print(result.argmax(1))
print(result.shape)

submission.to_csv(filepath + 'my_submission.csv', index=False)


# 0.9873 --> 0.9362745098