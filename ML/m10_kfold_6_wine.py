import warnings
warnings.filterwarnings('ignore')

#1. data (categorical classification)
import numpy as np
from sklearn.datasets import load_wine 
dataset = load_wine()
x = dataset.data 
y = dataset.target

#1-0. preprocessing
seed = 66
from sklearn.model_selection import train_test_split, KFold, cross_val_score
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=seed)
kfold = KFold(n_splits=5, shuffle=True, random_state=seed) 

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

# KNeighborsClassifier() scores :   [0.66 0.79 0.57 0.71 0.5 ]
# DecisionTreeClassifier() scores : [0.9  0.86 0.86 0.86 0.89]
# RandomForestClassifier() scores : [0.97 1.   0.93 0.93 0.96]
# LogisticRegression() scores :     [0.9  1.   0.82 0.89 0.96]
# LinearSVC() scores :              [0.93 0.66 0.86 0.89 0.71]
# SVC() scores :                    [0.59 0.66 0.5  0.68 0.68]