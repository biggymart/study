# 시계열 > 임베딩 레이어의 구성 배울 것임
# 나머지 자연어 처리는 알아서 배울 것

from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 진짜 맛있는 밥을 진짜 마구 마구 먹었다.'
# 어절별로 자르겠다

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)
# 워드의 순서, 빈도수가 높은 놈이 앞에 옴 (1번 인덱스)
# {'진짜': 1, '마구': 2, '나는': 3, '맛있는': 4, '밥을': 5, '먹었다': 6}

x = token.texts_to_sequences([text])
print(x)
# 수치화하면 이렇게 됨
# [[3, 1, 1, 4, 5, 1, 2, 2, 6]]
# 문제점: 그러면 '나는'은 '진짜'의 3배인가?
# 해결책: OneHotEncoding

from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)
print(word_size) # 6

x = to_categorical(x)
print(x)
# [[[0. 0. 0. 1. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0.]
#   [0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1.]]]
print(x.shape) # (1, 9, 7) 
# 문장 1개
# 단어 9개
# 단어종류: 7-1 == 6개 (인덱스의 시작점이 0 이라서 7이 나온 것)
