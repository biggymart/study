#1. data
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
dataset = load_boston()
x = dataset.data
y = dataset.target

#1-0. preprocessing
seed = 66
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = seed)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

#2. model
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
scaler1 = MinMaxScaler()
scaler2 = StandardScaler()
scaler_lst = [scaler1, scaler2]
for scaler in scaler_lst:
    model = make_pipeline(scaler, RandomForestRegressor())
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print(str(scaler), results)
# MinMaxScaler() 0.925197695099826
# StandardScaler() 0.921858099194192
