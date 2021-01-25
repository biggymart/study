from nltk.stem import WordNetLemmatizer
n = WordNetLemmatizer()
words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print([n.lemmatize(w) for w in words])
# ['policy', 'doing', 'organization', 'have', 'going', 'love', 'life', 'fly', 'dy', 'watched', 'ha', 'starting']
# lives -> life, dies -> dy, has -> ha
# 표제어 추출기(lemmatizer)가 본래 단어의 품사 정보를 알아야만 정확한 결과를 얻을 수 있기 때문, POS 태그를 보존 = 해당 단어의 품사 정보를 보존
# 단어의 형태가 적절히 보존됨

print(n.lemmatize('dies', 'v'))
# die
print(n.lemmatize('watched', 'v'))
# watch
print(n.lemmatize('has', 'v'))
# have