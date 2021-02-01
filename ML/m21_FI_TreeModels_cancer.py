# Task requirement:
# Reconstuct dataset by removing columns which FI is 0
# Use DesicionTree as model and check its accuracy_score

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

def plot_feature_importances_datasets(model, datasets):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel("Features")
    plt.ylim(-1, n_features)
    return plt

#1. data
dataset = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)

model = DecisionTreeClassifier(max_depth=4) # RandomForestClassifier, GradientBoostingClassifier

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df.drop(cut_columns(model.feature_importances_, dataset.feature_names, 4), axis=1, inplace=True)

#3. fit
model.fit(x_train, y_train, eval_metric='logloss')

#4. score and predict
acc = model.score(x_test, y_test)
print("acc : ",acc)

#5. visualization
plot_feature_importances_datasets(model, dataset)
plt.show()


### 실행해보고 결과 기록 ###

# DecisionTreeClassifier

# RandomForestClassifier

# GradientBoostingClassifier