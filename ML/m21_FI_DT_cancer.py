# Task requirement:
# Reconstuct dataset by removing columns which FI is 0
# Use DesicionTree as model and check its accuracy_score

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#1. data
dataset = load_breast_cancer()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

x_feature_names = ['worst perimeter', 'worst texture', 'worst radius']
### m24 함수 확인해보길 ###
df_trim = df[x_feature_names]

x = df_trim.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, dataset.target, train_size=0.8, random_state=44)

#2. model
model = DecisionTreeClassifier(max_depth=4)

#3. fit
model.fit(x_train, y_train)

#4. score and predict
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print("acc :", acc)


def plot_feature_importances_dataset(model):
    n_features = dataset.data.shape[1]
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

'''
def cut_columns(feature_importances, columns, number):
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
'''