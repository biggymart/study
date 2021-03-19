from tensorflow.keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)

sequences = tokenizer.texts_to_sequences(samples)
print(sequences)
# [[1, 2, 3, 4, 1, 5], [1, 6, 7, 8, 9]]

one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
print(one_hot_results.shape)

word_index = tokenizer.word_index
print(word_index)