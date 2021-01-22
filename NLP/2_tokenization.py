# Tokenization


s1 = "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."
from nltk.tokenize import word_tokenize
print(word_tokenize(s1))
# ['Do', "n't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr.', 'Jone', "'s", 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']

from nltk.tokenize import WordPunctTokenizer  
print(WordPunctTokenizer().tokenize(s1))
# ['Don', "'", 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr', '.', 'Jone', "'", 's', 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']

from tensorflow.keras.preprocessing.text import text_to_word_sequence
print(text_to_word_sequence(s1))
# ["don't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', 'mr', "jone's", 'orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop']


# Standard word tokenization
from nltk.tokenize import TreebankWordTokenizer
tokenizer=TreebankWordTokenizer()
# 규칙 1. 하이푼으로 구성된 단어는 하나로 유지한다.
# 규칙 2. doesn't와 같이 아포스트로피로 '접어'가 함께하는 단어는 분리해준다.

s2 ="Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
print(tokenizer.tokenize(s2))
# ['Starting', 'a', 'home-based', 'restaurant', 'may', 'be', 'an', 'ideal.', 'it', 'does', "n't", 'have', 'a', 'food', 'chain', 'or', 'restaurant', 'of', 'their', 'own', '.']


# Sentence tokenization
from nltk.tokenize import sent_tokenize
s3 = "His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
print(sent_tokenize(s3))
# ['His barber kept his word.', 'But keeping such a huge secret to himself was driving him crazy.', 'Finally, the barber went up a mountain and almost to the edge of a cliff.', 'He dug a hole in the midst of some reeds.', 'He looked about, to make sure no one was near.']

from nltk.tokenize import sent_tokenize
s4 = "I am actively looking for Ph.D. students. and you are a Ph.D student."
print(sent_tokenize(s4))
# ['I am actively looking for Ph.D. students.', 'and you are a Ph.D student.']

# pip install kss
import kss
s5 = '딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어려워요. 농담아니에요. 이제 해보면 알걸요?'
print(kss.split_sentences(s5))
# ['딥 러닝 자연어 처리가 재미있기는 합니다.', '그런데 문제는 영어보다 한국어로 할 때 너무 어려워요.', '농담아니에요.', '이제 해보면 알걸요?']

# abbreviation dictionary
# https://public.oed.com/how-to-use-the-oed/abbreviations/
# NLTK, OpenNLP, 스탠포드 CoreNLP, splitta, LingPipe 
# https://www.grammarly.com/blog/engineering/how-to-split-sentences/ --> OntoNotes and MASC corpora


# 몇 가지 개념들:
# 1. 형태소(morpheme)
# 자립 형태소 vs 의존 형태소 
# 2. 품사 태깅(part-of-speech tagging)

from nltk.tag import pos_tag
x = word_tokenize(s4)
print(pos_tag(x))
# [('I', 'PRP'), ('am', 'VBP'), ('actively', 'RB'), ('looking', 'VBG'), ('for', 'IN'), ('Ph.D.', 'NNP'), ('students', 'NNS'), ('.', '.'), ('and', 'CC'), ('you', 'PRP'), ('are', 'VBP'), ('a', 'DT'), ('Ph.D', 'NNP'), ('student', 'NN'), ('.', '.')]
# PRP는 인칭 대명사, VBP는 동사, RB는 부사, VBG는 현재부사, IN은 전치사, NNP는 고유 명사, NNS는 복수형 명사, CC는 접속사, DT는 관사

# KoNLPy: 사용할 수 있는 형태소 분석기로 Okt(Open Korea Text), 메캅(Mecab), 코모란(Komoran), 한나눔(Hannanum), 꼬꼬마(Kkma)
from konlpy.tag import Okt  
okt = Okt()  
print(okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))