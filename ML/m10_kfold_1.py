import warnings
warnings.filterwarnings('ignore')

#1. data (categorical classification)
import numpy as np
from sklearn.datasets import load_iris
dataset = load_iris()
x = dataset.data # (150, 4)
y = dataset.target # (150, )

#1-0. preprocessing
seed = 66
from sklearn.model_selection import KFold, cross_val_score
kfold = KFold(n_splits=5, shuffle=True, random_state=seed) ### Takeaway1 ### 전체 데이터에서 5등분

#2. model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression # (이진분류)
from sklearn.svm import LinearSVC, SVC

m1 = KNeighborsClassifier()
m2 = DecisionTreeClassifier()
m3 = RandomForestClassifier()
m4 = LogisticRegression()
m5 = LinearSVC()           
m6 = SVC()
model_lst = [m1, m2, m3, m4, m5, m6]

#3. fit
#4. score
np.set_printoptions(precision=2)
for model in model_lst:
    scores = cross_val_score(model, x, y, cv=kfold) ### Takeaway2 ###
    print('{0} scores : {1}'.format(str(model), scores))
# 모델과 데이터가 엮이는 지점, fit과 score 모두 포함되어 있음

# KNeighborsClassifier() scores :   [0.97 0.97 1.   0.9  0.97]
# DecisionTreeClassifier() scores : [0.93 0.97 1.   0.9  0.93]
# RandomForestClassifier() scores : [0.97 0.97 1.   0.9  0.97]
# LogisticRegression() scores :     [1.   0.97 1.   0.9  0.97]
# LinearSVC() scores :              [0.97 0.97 1.   0.9  1.  ]
# SVC() scores :                    [0.97 0.97 1.   0.93 0.97]