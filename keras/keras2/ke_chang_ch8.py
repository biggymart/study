import keras
import numpy as np

path = keras.utils.get_file(
    'nietzsche.txt',
    origin = 'https://s3.amazonaws.com/text-datasets/nietzsche.txt'
)
text = open(path).read().lower()
# print(text)
print('말뭉치 크기:', len(text)) # 600893

maxlen = 60
step = 3

sentences = []
next_chars = []


for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

# print(sentences)
print('시퀀스 개수:', len(sentences)) # 200278

chars = sorted(list(set(text))) # 말뭉치에서 고유한 글자를 담은 리스트
print('고유한 글자:', len(chars)) # 58
char_indices = dict((char, chars.index(char)) for char in chars)
# print(char_indices)
# {'\n': 0, ' ': 1, '!': 2, '"': 3, (중략), '횈': 57}

print('벡터화...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool) # (200278, 60, 58)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool) # (200278, 58)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

from keras import layers

model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))

optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

import random
import sys

random.seed(42)
start_index = random.randint(0, len(text) - maxlen - 1)

for epoch in range(1, 60):
    print('에포크', epoch)
    model.fit(x, y, batch_size=128, epochs=1)

    seed_text = text[start_index: start_index + maxlen]
    print(' --- 시드 텍스트: "' + seed_text + '"')

    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print(' ------- 온도:', temperature)
        generated_text = seed_text
        sys.stdout.write(generated_text)

        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.

            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:]

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()