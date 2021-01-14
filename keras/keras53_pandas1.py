import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()
print(dataset.keys()) 
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
print(dataset.values())

print(dataset.target_names)
# ['setosa' 'versicolor' 'virginica']

# x = dataset.data
x = dataset['data'] # 딕셔너리 형식
# y = dataset.target
y = dataset['target']

print(x, y, x.shape, y.shape) # (150, 4) (150,) # 리스트는 shape이 안 먹힌다
print(type(x), type(y)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

df = pd.DataFrame(dataset.data, columns=dataset.feature_names) # numpy를 pandas로 바꾸기, dataset['target']
print(df) # [150 rows x 4 columns] # header와 index는 데이터가 아니다
print(df.shape) # (150, 4)
print(df.columns) # Index(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)'], dtype='object')
print(df.index) # RangeIndex(start=0, stop=150, step=1) # 인덱스가 명시되지 않으면 자동인덱싱

print(df.head()) # 앞에 일부분만 보여줌
print(df.tail()) # 뒤에 일부분만 보여줌
print(df.info()) # non-null 결측치가 없다
print(df.describe()) #

# column 명을 짧게 해주자
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
print(df.columns)
print(df.info())
print(df.describe())

# y 칼럼을 추가해 보아요
print(df['sepal_length'])
df['target'] = dataset.target
print(df.head())
print(df.shape) # (150, 5)
print(df.columns) # target 추가됨
print(df.index) # 동일

print(df.isnull()) # False 이면 non-null이라느 것
print(df.isnull().sum()) # 각 column에서 결측치 갯수 더해서 보여줌
print(df.describe()) # 평균, 표준편차 등 정보 제공
print(df['target'].value_counts()) # y값에 몇 개씩 있는지 세어줌

# 상관계수 correlation coefficient (Karl Pearson)
# 두 변수 사이의 상관관계의 정도를 나타내는 수치, linear하게 계산한 결과
# https://leedakyeong.tistory.com/
print(df.corr()) # 각 변수 사이의 상관계수를 나타내줌, feature engineering 할 때 추가 및 제거하게 됨

# 시각화
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True) # 사각형 형태, annotation 글씨 넣어주기, column bar
plt.show()

# 도수 분포표 (histogram)
plt.figure(figsize=(10, 6)) # 도화지 준비

plt.subplot(2, 2, 1)
plt.hist(x='sepal_length', data=df)
plt.title('sepal_length')

plt.subplot(2, 2, 2)
plt.hist(x='sepal_width', data=df)
plt.title('sepal_width')

plt.subplot(2, 2, 3)
plt.hist(x='petal_length', data=df)
plt.title('petal_length')

plt.subplot(2, 2, 4)
plt.hist(x='petal_width', data=df)
plt.title('petal_width')

plt.show()

'''
.index
.columns
.head()
.tail()
.shape()
.info()
.describe()
.isnull()
.isnull().sum()
.valuecount()
.corr()
'''