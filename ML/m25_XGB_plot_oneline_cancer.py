from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance ### Takeaway1 ###
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def cut_columns(feature_importances,columns,number):
    temp = []
    print(len(feature_importances))
    for i in feature_importances:
        temp.append(i)
    temp.sort()
    temp=temp[:number]
    result = []
    for j in temp:
        index = feature_importances.tolist().index(j)
        result.append(columns[index])
    return result

#1. data
datasets = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(datasets.data, datasets.target, test_size=0.15)

#2. model
model = XGBClassifier(n_jobs = -1)

#3. train
model.fit(x_train, y_train)
y = datasets.target

#4. score and predict
acc = model.score(x_test,y_test)
print(model.feature_importances_)
print(datasets.feature_names)
print("acc : ",acc)

df = pd.DataFrame(datasets.data,columns=datasets.feature_names)
df.drop(cut_columns(model.feature_importances_,datasets.feature_names,4),axis=1,inplace=True)
print(cut_columns(model.feature_importances_,datasets.feature_names,4))
x_train,x_test,y_train,y_test = train_test_split(df.values,datasets.target,test_size=0.15)

# 훈련
model.fit(x_train,y_train)
y = datasets.target
# 평가, 예측
acc = model.score(x_test,y_test)
print("acc : ",acc)

plot_importance(model) ### Takeaway2 ###
# m20_plot_FI1_iris.py 와 비교; 한 줄로 처리 가능
# x axis: f score
plt.show()
