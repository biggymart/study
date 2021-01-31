# FI가 0인 칼럼을 제거하여 데이터셋을 재구성
# DesicionTree로 모델을 돌려서 acc를 확인해보시오

from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd

#1. data
dataset = load_boston()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
# print(dataset.feature_names)

x_feature_names = ['LSTAT', 'DIS', 'RM']
df_trim = df[x_feature_names]

x = df_trim.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, dataset.target, train_size=0.8, random_state=44)

#2. model
model = DecisionTreeRegressor(max_depth=4)

#3. fit
model.fit(x_train, y_train)

#4. score and predict
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print("acc :", acc)

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model):
    n_features = x.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), x_feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()

# [0.23240953 0.12087774 0.64671272]
# acc : 0.7723676078931413