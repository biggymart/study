# 이 종류에서는 끝판왕 급이지만 어려울 건 없다
# 전처리까지 합치는 걸 pipeline이라고 한다

#1. data
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
dataset = load_iris()
x = dataset.data
y = dataset.target

#1-0. preprocessing
seed = 66
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = seed)

# scaler = MinMaxScaler()
# scaler.fit_transform(x_train)
# scaler.transform(x_test)
# 아래 pipeline에서 엮어주기 때문에 할 필요 없음

#2. model
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline # 조금 다른 철자/문법일 뿐 용도는 같음, concatenate과 비슷
# model = Pipeline([("scaler", MinMaxScaler()), ('model_tobepiped', SVC())]) # 이름을 부여할 뿐 make_pipine과 다를 것 없음
# 전처리 하나와 모델 한 개만 엮어준 것 ### Takeaway1 ###
model = make_pipeline(StandardScaler(), SVC())
model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print(results)

# MinMaxScaler
# 1.0
# StandardScaler
# 0.9333333333333333