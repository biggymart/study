# 데이터는 wine
# 모델은 RandomForest 사용
# 파이프라인 엮어서 25번 돌리기!

import warnings
warnings.filterwarnings('ignore')

#1. data
import numpy as np
from sklearn.datasets import load_wine
dataset = load_wine()
x = dataset.data
y = dataset.target

#1-0. preprocessing
seed = 66
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = seed)
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

# dictionary within list
parameters = [
    {"aaa__n_estimators" : [100, 200], "aaa__max_depth" : [6, 8, 10, 12], "aaa__min_samples_leaf" : [3, 5, 7, 10], 
    "aaa__min_samples_split" : [2, 3, 5, 10], "aaa__n_jobs" : [-1]}
]

#2. model
#3. fit #4. score (cross_val_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler1 = MinMaxScaler()
scaler2 = StandardScaler()
scaler_lst = [scaler1, scaler2]

for scaler in scaler_lst:
    pipe1 = Pipeline([("scaler", scaler), ('aaa', RandomForestClassifier())])
    model1 = GridSearchCV(pipe1, parameters, cv=kfold)
    results1 = cross_val_score(model1, x_train, y_train, cv=kfold)
    print("GridSearchCV")
    print(str(scaler), results1)

for scaler in scaler_lst:
    pipe2 = Pipeline([("scaler", scaler), ('aaa', RandomForestClassifier())])
    model2 = RandomizedSearchCV(pipe2, parameters, cv=kfold)
    results2 = cross_val_score(model2, x_train, y_train, cv=kfold)
    print("RandomizedSearchCV")
    print(str(scaler), results2)

# GridSearchCV
# MinMaxScaler() [0.93103448 1.         0.92857143 0.89285714 0.96428571]
# GridSearchCV
# StandardScaler() [0.96551724 1.         0.92857143 0.96428571 0.96428571]
# RandomizedSearchCV
# MinMaxScaler() [0.96551724 0.93103448 0.92857143 0.89285714 0.96428571]
# RandomizedSearchCV
# StandardScaler() [0.96551724 0.96551724 0.92857143 0.92857143 0.96428571]