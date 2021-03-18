from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
)

print(x_train[0], type(x_train[0]))
print(y_train[0])
print(len(x_train[0]), len(x_train[11]))
# 글자 하나하나를 수치화한 것
# [1, 2, 2, 8, 43, 10, 447, (중략) 6, 109, 15, 17, 12] <class 'list'>
# 3      # 어떤 한 카테고리
# 87 59  # 길이가 다른 기사

print("=====================================")
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
print("뉴스기사 최대길이 :", max(len(l) for l in x_train))
print("뉴스기사 평균길이 :", sum(map(len, x_train))/ len(x_train))
# (8982,) (2246,)
# (8982,) (2246,)
# 뉴스기사 최대길이 : 2376
# 뉴스기사 평균길이 : 145.5398574927633

# x_train 길이 히스토그램
plt.hist([len(s) for s in x_train], bins=50)
plt.show()

unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("y 분포 :", dict(zip(unique_elements, counts_elements)))
# y_train 분포 히스토그램
plt.hist(y_train, bins=46)
plt.show()
# y 분포 : {0: 55, 1: 432, 2: 74, 3: 3159, 4: 1949, (중략) 45: 18}
print("=====================================")

# x의 단어들 분표
word_to_index = reuters.get_word_index()
print(word_to_index)
print(type(word_to_index))
# {'mdbl': 10996, 'fawc': 16260, 'degussa': 12089, 'woods': 8803, 'hanging': 13796, (중략)} 
# 이게 input_dim (혹은 word_size)이 되는 것
# <class 'dict'>
print("=====================================")

# 원문을 확인하기 위해서 키와 밸류를 교체
index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value] = key
print(index_to_word)
print(index_to_word[1]) # the
print(index_to_word[30979])
print(len(index_to_word))

# x_train[0]
print(' '.join([index_to_word[index] for index in x_train[0]]))
# num_words = 10000 이라서 문장이 10000개 순위 내에 드는 것만 가져와서 문장이 이상함
# num_words = 30000 로 하면 문장이 좀 더 자연스러움

# y 카테고리 개수 출력
category = np.max(y_train) + 1
print("y 카테고리 개수:", category) # 46

# y의 유니크한 값 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

################################### 전처리 ######################################
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, maxlen=100, padding='pre')
x_test = pad_sequences(x_test, maxlen=100, padding='pre')
print(x_train.shape, x_test.shape)
# (8982, 100) (2246, 100)

# sparse_categorical_crossentropy 활용할 시 주석 처리
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(y_train.shape)
# (8982, 46)
#################################################################################

# 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv1D, Flatten
model = Sequential()
# model.add(Embedding(input_dim=10000, output_dim=64, input_length=100)) # input_dim == num_words, input_length == maxlen
model.add(Embedding(10000, 64))
model.add(LSTM(32))
model.add(Dense(46, activation='softmax'))

# model.summary()

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)

results = model.evaluate(x_test, y_test)
print('loss :', results[0])
print('acc :', results[1])


