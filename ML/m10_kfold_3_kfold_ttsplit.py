# train_test_split 이후에 kfold 대신 (m10_kfold_2.py),
# kfold 한 후에 train_test_split 사용
import warnings
warnings.filterwarnings('ignore')

#1. data (categorical classification)
import numpy as np
from sklearn.datasets import load_iris
dataset = load_iris()
x = dataset.data
y = dataset.target

#1-0. preprocessing
# KFold.split : 데이터를 학습 및 테스트 세트로 분할하는 인덱스를 생성
seed = 66
from sklearn.model_selection import train_test_split, KFold, cross_val_score
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
for train_index, test_index in kfold.split(x):
    # train : test
    x_train, x_test = x[train_index], x[test_index] 
    y_train, y_test = y[train_index], y[test_index]
      
    # train : test : validation
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = seed, shuffle=True) 

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
    scores = cross_val_score(model, x_train, y_train, cv=kfold)
    print('{0} scores : {1}'.format(str(model), scores))

# KNeighborsClassifier() scores :   [0.9  0.95 1.   0.84 0.89]
# DecisionTreeClassifier() scores : [0.9  0.95 1.   0.95 0.89]
# RandomForestClassifier() scores : [0.9  0.95 1.   0.89 0.95]
# LogisticRegression() scores :     [0.9  0.95 1.   0.89 1.  ]
# LinearSVC() scores :              [0.95 0.89 1.   0.89 1.  ]
# SVC() scores :                    [0.8  0.95 1.   0.89 0.89]