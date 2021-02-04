# 인트로 썰풀기
# GridSearch 모든 것을 다 돌리기 때문에 느림, DL에서 dropout 처럼 과적합이 예방되었듯이
# 하지만 다 돌릴 필요 없음; RandomSearch는 그 중 일부만 본다 (표본집단 vs 모집단)
# 그리고 너네들이 준 파라미터가 신뢰도가 높은 거냐? 아니다, 감성적인 파라미터잖냐
# 파라미터는 still 개발자가 설정해주어야 하고, randomized는 그저 그 중 파라미터 몇 개를 빼서 계산해주는 것임 (dropout 기능 비슷)

# sidetracked story...
# ML 모든 모델을 본 것은 아니지만 DecisionTree, RandomForest '나무를 보지 말고 숲을 보라' '숲도 보지 말고 XGBooster를 봐라'
# sample split : 2번짜르는 갯수 (경우의 수 수형도), leaf인가 헷갈리는군

# 과제 Q: RandomizedSearchCV() 에서 랜덤하게 샘플링하는 크기가 어느정도냐?
# A: 각 fold 마다 sampling 하는 것은, n_iter 파라미터에 명시해주지 않는 경우, 10입니다.
# 이하 설명:
# https://scikit-learn.org/stable/modules/grid_search.html#randomized-parameter-optimization
# sklearn 공식문서에서
# Additionally, a computation budget, being the number of sampled candidates or sampling iterations, is specified using the n_iter parameter. For each parameter, either a distribution over possible values or a list of discrete choices (which will be sampled uniformly) can be specified:
# 이라고 명시되어 있고, n_iter의 디폴트 값은 10이기 때문에, 따라서 각 fold 마다 sampling 하는 것은, n_iter 파라미터에 명시해주지 않는 경우, 10입니다.

import timeit # 시간측정

import warnings
warnings.filterwarnings('ignore')

#1. data
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
# Refer to 'keras53_pandas2_to_csv.py' to see how a file is saved as a csv file

dataset = pd.read_csv(filepath_or_buffer='../data/csv/iris_sklearn.csv', header=0, index_col=0)
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

#1-0. preprocessing
seed = 66
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV ### Takeaway1 ###
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = seed)
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

# dictionary within list
parameters = [
    {"C" : [1, 10, 100, 1000], "kernel" : ["linear"]}, 
    {"C" : [1, 10, 100], "kernel" : ["rbf"], "gamma" : [0.001, 0.0001]}, 
    {"C" : [1, 10, 100, 1000], "kernel" : ["sigmoid"], "gamma" : [0.001, 0.0001]}
] ### Takeaway2 ###

#2. model
from sklearn.svm import SVC
model = RandomizedSearchCV(SVC(), parameters, cv=kfold) ### Takeaway3 ###

#3. fit
start_time = timeit.default_timer() # 시작 시간 체크
model.fit(x_train, y_train)
terminate_time = timeit.default_timer() # 종료 시간 체크


#4. score
print('최적의 매개변수: {0}'.format(model.best_estimator_)) ### Takeaway4 ###

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print('최종정답률: {0}'.format(accuracy_score(y_test, y_pred))) ### Takeaway5 ###
# Alternative: model.score(x_test, y_test) 
print("%f초 걸렸습니다." % (terminate_time - start_time))

# 최적의 매개변수: SVC(C=1, kernel='linear')
# 최종정답률: 0.9666666666666667

# 어맛! 빠르다!
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
# 시간 비교하는 공식문서 사이트
# https://www.geeksforgeeks.org/timeit-python-examples/
# timeit 사용법