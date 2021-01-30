import warnings
warnings.filterwarnings('ignore')

#1. data
import numpy as np
from sklearn.datasets import load_iris
dataset = load_iris()
x = dataset.data
y = dataset.target

#1-0. preprocessing
seed = 66
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = seed)
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

# dictionary within list
parameters = [
    {"C" : [1, 10, 100, 1000], "kernel" : ["linear"]}, 
    {"C" : [1, 10, 100], "kernel" : ["rbf"], "gamma" : [0.001, 0.0001]}, 
    {"C" : [1, 10, 100, 1000], "kernel" : ["sigmoid"], "gamma" : [0.001, 0.0001]}
]

#2. model
#3. fit
#4. score
from sklearn.svm import SVC
model = GridSearchCV(SVC(), parameters, cv=kfold) 
# 최적값이 하나 나오고
score = cross_val_score(model, x_train, y_train, cv=kfold)
# 그걸 5개로 나눠서 또 돌림
# 총 25번 돌아감

print("교차검증점수 :", score)

# 교차검증점수 : [1.         0.95833333 0.95833333 1.         1.        ]