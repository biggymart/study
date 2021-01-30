#1. data
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
dataset = load_boston()
x = dataset.data
y = dataset.target

#1-0. preprocessing
seed = 66
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = seed)
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

parameters1 = [{"aaa__n_estimators" : [100, 200], "aaa__max_depth" : [6, 8, 10, 12], "aaa__min_samples_leaf" : [3, 5, 7, 10], "aaa__min_samples_split" : [2, 3, 5, 10], "aaa__n_jobs" : [-1, 2, 4]}]
parameters2 = [{"randomforestregressor__n_estimators" : [100, 200], "randomforestregressor__max_depth" : [6, 8, 10, 12], "randomforestregressor__min_samples_leaf" : [3, 5, 7, 10], "randomforestregressor__min_samples_split" : [2, 3, 5, 10], "randomforestregressor__n_jobs" : [-1, 2, 4]}]

#2. model
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
scaler1 = MinMaxScaler()
scaler2 = StandardScaler()
scaler_lst = [scaler1, scaler2]
for scaler in scaler_lst:
    pipe1 = Pipeline([("scaler", scaler), ('aaa', RandomForestRegressor())])
    model1 = GridSearchCV(pipe1, parameters1, cv=kfold)
    model1.fit(x_train, y_train)
    results = model1.score(x_test, y_test)
    print("GridSearchCV")
    print(str(scaler), results)

for scaler in scaler_lst:
    pipe2 = make_pipeline(scaler, RandomForestRegressor())
    model2 = RandomizedSearchCV(pipe2, parameters2, cv=kfold)
    model2.fit(x_train, y_train)
    results = model2.score(x_test, y_test)
    print("RandomizedSearchCV")
    print(str(scaler), results)


