# 긍정과 부정을 맞춰보자
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

# X data
docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화예요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요', '규현이가 잘 생기긴 했어요'
]

# y data: 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

# 전처리
token = Tokenizer()
token.fit_on_texts(docs)
x = token.texts_to_sequences(docs)

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=4) # 'post'
pad_x = np.reshape(pad_x, (pad_x.shape[0], pad_x.shape[1], 1))
# print(pad_x.shape) # (13, 4, 1)

# modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D

# 임베딩 레이어 없는 모델
model = Sequential()
model.add(LSTM(32, input_shape=(4, 1)))
model.add(Flatten()) 
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, epochs=100)

acc = model.evaluate(pad_x, labels)[1] 
# loss, metrics를 반환하므로 1인덱스는 'acc'임
print(acc)