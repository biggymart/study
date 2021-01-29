# 다중연결 pipeline + GridSearchCV
# cross_validation 할 때마다 train set만 scaling 해주기 위해 pipeline이 만들어졌다. (map, zip 비슷)

# 데이터 전처리는 train set만 하는 게 더 효율적(참조> keras18_boston4_MinMaxScaler.py)이기 때문에 pipeline을 사용하는 것
# 전체 데이터에 대해서 전처리하면 과적합의 문제가 발생

#1. data
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
dataset = load_iris()
x = dataset.data
y = dataset.target

#1-0. preprocessing
seed = 66
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = seed)
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

# parameters = [
#     {"aaa__C" : [1, 10, 100, 1000], "aaa__kernel" : ["linear"]}, 
#     {"aaa__C" : [1, 10, 100], "aaa__kernel" : ["rbf"], "aaa__gamma" : [0.001, 0.0001]}, 
#     {"aaa__C" : [1, 10, 100, 1000], "aaa__kernel" : ["sigmoid"], "aaa__gamma" : [0.001, 0.0001]}
# ] ### Takeaway1, Pipeline ###

parameters = [
    {"svc__C" : [1, 10, 100, 1000], "svc__kernel" : ["linear"]}, 
    {"svc__C" : [1, 10, 100], "svc__kernel" : ["rbf"], "svc__gamma" : [0.001, 0.0001]}, 
    {"svc__C" : [1, 10, 100, 1000], "svc__kernel" : ["sigmoid"], "svc__gamma" : [0.001, 0.0001]}
] ### make_pipeline ###

#2. model
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
scaler1 = MinMaxScaler()
scaler2 = StandardScaler()
scaler_lst = [scaler1, scaler2]
for scaler in scaler_lst:
    # pipe =  Pipeline([("scaler", scaler), ('aaa', SVC())]) ### Takeaway2 ###
    pipe = make_pipeline(StandardScaler(), SVC()) # parameters 변경해줘야 함
    model = GridSearchCV(pipe, parameters, cv=kfold)
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print(str(scaler), results)
# Total cycle: kfold * parameters * scalers

# MinMaxScaler() 0.9666666666666667
# StandardScaler() 1.0