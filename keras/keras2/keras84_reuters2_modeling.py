from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
)

print(type(x_train))
print(x_train.shape)

# 전처리
# maxlen을 몇으로 정할 것인가?
pad_x_train = pad_sequences(x_train, padding='pre', maxlen=500)
pad_x_test = pad_sequences(x_test, padding='pre', maxlen=500)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# modeling
model = Sequential()
# output_dim과 input_length를 몇으로 정할 것인가?
model.add(Embedding(input_dim=10000, output_dim=120, input_length=500))
model.add(Dropout(rate=0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate=0.5))
# model.add(LSTM(32))
model.add(Dense(46, activation='softmax'))

model.summary()

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
re = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit(pad_x_train, y_train, epochs=100, validation_data=(pad_x_test, y_test), callbacks=[es, re])

acc = model.evaluate(pad_x_test, y_test) 
# loss, metrics를 반환하므로 1인덱스는 'acc'임
print(acc)

# 참고
# https://wikidocs.net/22933
