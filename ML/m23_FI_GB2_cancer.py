# boost 계열 중 GradientBoost
# column 정리해서 돌려서 비교 (col 정리 전후, DecisionTree, RandomForest)

from sklearn.ensemble import GradientBoostingClassifier ### Takeaway1 ###
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd

#1. data
dataset = load_breast_cancer()
# 랜덤포레스트로 결과치 보고 하위 25% 피쳐 줄여서 만들기
# 힌트: 결과를 sort해라
'''
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
x_feature_names = ['worst perimeter', 'worst texture', 'worst radius']
df_trim = df[x_feature_names]
x = df_trim.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, dataset.target, train_size=0.8, random_state=44)
'''
#2. model
model = GradientBoostingClassifier()

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

# 이전 결과
# [0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.00677572 0.         0.         0.         0.00677572 0.
#  0.         0.         0.         0.05612587 0.78000877 0.01008994
#  0.02293065 0.         0.         0.11729332 0.         0.        ]
# acc : 0.9385964912280702

# 이후 결과
# [0.88905842 0.09991873 0.01102285]
# acc : 0.9473684210526315