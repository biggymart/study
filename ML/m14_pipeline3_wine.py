#1. data
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
dataset = load_wine()
x = dataset.data
y = dataset.target

#1-0. preprocessing
seed = 66
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = seed)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

#2. model
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
scaler1 = MinMaxScaler()
scaler2 = StandardScaler()
scaler_lst = [scaler1, scaler2]
for scaler in scaler_lst:
    model = make_pipeline(scaler, RandomForestClassifier())
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print(str(scaler), results)
# MinMaxScaler() 1.0
# StandardScaler() 1.0
