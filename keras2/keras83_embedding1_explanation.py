# 긍정과 부정을 맞춰보자
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화예요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요', '규현이가 잘 생기긴 했어요'
]

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '잘': 3, '재밌어요': 4, '최고에요': 5,
#  '만든': 6, '영화예요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10,
#  '한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15, '글쎄요': 16,
#  '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21,
#  '재미없어요': 22, '재미없다': 23, '재밌네요': 24, '규현이가': 25, '생기긴': 26, '했어요': 27}

# 문장의 수치화
x = token.texts_to_sequences(docs)
print(x)
# [[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16],
#  [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 3, 26, 27]]

# 문제점: 문장의 길이가 각각 다름
# 해결책: 긴 문장 기준으로 짧은 문장은 0을 채워줌

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5) # 'post'

# pad_x = pad_sequences(x, padding='pre')
# print(pad_x)
# print(pad_x.shape)
# [[ 0  0  0  2  4]
#  [ 0  0  0  1  5]
#  [ 0  1  3  6  7]
#  [ 0  0  8  9 10]
#  [11 12 13 14 15]
#  [ 0  0  0  0 16]
#  [ 0  0  0  0 17]
#  [ 0  0  0 18 19]
#  [ 0  0  0 20 21]
#  [ 0  0  0  0 22]
#  [ 0  0  0  2 23]
#  [ 0  0  0  1 24]
#  [ 0 25  3 26 27]]
# (13, 5)

# 일반적으로 시계열에서 뒷쪽에 있을 수록 더 영향을 많이 미침
# (13, 5, 1) 로 바꾸면 LSTM 가능

# 동일한 길이로 짜르는 파라미터
# 예를 들어> 문장 길이를 일정하게 4로 자르고 싶다
# maxlen=4, truncating='pre'
# 예를 들어> 주식의 월별 일자를 30으로 일정하게 자르고 싶다
# maxlen=30, truncating='post'
'''
# pad_x = pad_sequences(x, padding='pre', maxlen=4)
print(pad_x)
print(pad_x.shape)
# [[ 0  0  2  4]
#  [ 0  0  1  5]
#  [ 1  3  6  7]
#  [ 0  8  9 10]
#  [12 13 14 15]
#  [ 0  0  0 16]
#  [ 0  0  0 17]
#  [ 0  0 18 19]
#  [ 0  0 20 21]
#  [ 0  0  0 22]
#  [ 0  0  2 23]
#  [ 0  0  1 24]
#  [25  3 26 27]]
# (13, 4)

print(np.unique(pad_x))
# [ 0  1  2  3  4  5  6  7  8  9 10 12 13 14 15 16 17 18 19 20 21 22 23 24
#  25 26 27]
# 11이 짤려서 없는 것을 확인할 수 있다
print(len(np.unique(pad_x))) 
# 27
# 원래 0부터 27인데 11이 빠져서 27개
'''
# modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D

model = Sequential()
# 문제점: OneHotEncoding을 하게 되면 너무 큼
#        만약에 단어 종류가 100만개 된다면 1:1 대응으로 크기가 그만큼 커짐;;
# 해결책: 벡터화, 차원축소, 거리별로 계산
model.add(Embedding(input_dim=28, output_dim=11, input_length=5))
# 임베딩은 특별하게 인풋을 가장 먼저 받음
# input_dim을 word_size라고도 함, 총 단어사전의 개수 (최적은 일치하는 것)
# BUT input_dim은 단어사전의 개수보다 커도 되지만 (연산량은 커짐)
#     단어사전의 개수보다 작으면 오류
# 정리하자면, input_dim >= word_size 여야 함

# 아웃풋 딤은 임의로 정함, 다음 레이어로 전달해주는 노드의 개수 (벡터화, 11로 압축) cf> PCA 와 유사
# input_length 데이터 구조 마지막 컬럼

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, 5, 11)             308
# _________________________________________________________________
# flatten (Flatten)            (None, 55)                0
# _________________________________________________________________
# dense (Dense)                (None, 1)                 56
# =================================================================
# Total params: 364
# Trainable params: 364
# Non-trainable params: 0
# _________________________________________________________________

'''
# Side note:
model.add(Embedding(28, 11))
model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, None, 11)          308
# =================================================================
# Total params: 308
# Trainable params: 308
# Non-trainable params: 0
# _________________________________________________________________
# 즉, input_length 파라미터를 넣지 않고도 돌아가긴 함

# Embedding layer의 출력이 3차원이므로 바로 LSTM에 넣을 수 있음
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, None, 11)          308
# _________________________________________________________________
# lstm (LSTM)                  (None, 32)                5632
# _________________________________________________________________
# dense (Dense)                (None, 1)                 33
# =================================================================
# Total params: 5,973
# Trainable params: 5,973
# Non-trainable params: 0
# _________________________________________________________________
'''

# Param #: 308 == 28 (단어사전 개수) * 11 (output_dim)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, epochs=100)

acc = model.evaluate(pad_x, labels)[1] 
# loss, metrics를 반환하므로 1인덱스는 'acc'임
print(acc)

