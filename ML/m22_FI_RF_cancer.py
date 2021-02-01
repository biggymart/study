# model: RandomForest, reduce lowest 25% features

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_feature_importances_dataset(model):
    n_features = x.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)
# plot_feature_importances_dataset(model)
# plt.show()


#1. data
dataset = load_breast_cancer()

# 힌트: 결과를 sort해라

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
# x_feature_names = ['worst perimeter', 'worst texture', 'worst radius']
# df_trim = df[x_feature_names]
x = df.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, dataset.target, train_size=0.8, random_state=44)

#2. model
model = RandomForestClassifier(max_depth=4)

#3. fit
model.fit(x_train, y_train)

#4. score and predict
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print(type(model.feature_importances_)) # <class 'numpy.ndarray'>
print(len(model.feature_importances_)//4) # 30//4 # How many

print("acc :", acc)


# [0.03418279 0.01576305 0.05031784 0.03062761 0.00409702 0.01678361
#  0.06265407 0.11523154 0.00247018 0.00247695 0.00579141 0.00220867
#  0.00511396 0.04302403 0.00291319 0.00451524 0.00281531 0.00606207
#  0.00266726 0.0026441  0.11158764 0.01467097 0.14303296 0.11876765
#  0.01756259 0.0134522  0.03051694 0.11599532 0.01525499 0.00679884] 
# FI의 값이 아니라 이름을 반환하는 건 없나?
# 하위 25%에 해당하는 피처가 무엇인지 알아내는 과정? 


# https://towardsdatascience.com/extracting-plotting-feature-names-importance-from-scikit-learn-pipelines-eb5bfa6a31f4
