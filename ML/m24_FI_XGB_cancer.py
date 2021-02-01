from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

#1. data
#2. model
datasets = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(datasets.data, datasets.target, test_size=0.2)

model = XGBClassifier(n_jobs = -1, use_label_encoder=False)

df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
df.drop(cut_columns(model.feature_importances_, datasets.feature_names, 4), axis=1, inplace=True)
print(cut_columns(model.feature_importances_, datasets.feature_names, 4))

#3. fit
model.fit(x_train, y_train, eval_metric='logloss')
y = datasets.target

#4. score and predict
acc = model.score(x_test, y_test)
print("acc : ",acc)

#5. visualization
plot_feature_importances_datasets(model, datasets)

# n_jobs : -1   # 신뢰가 가지 않는군
# 0:00:00.420282
# n_jobs : 8
# 0:00:00.416464
# n_jobs : 4
# 0:00:00.427209
# n_jobs : 1
# 0:00:00.464287