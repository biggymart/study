# m10_kfold_2.py와 동일
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
from sklearn.model_selection import train_test_split, KFold, cross_val_score
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=seed) ### Takeaway1 ### 
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
    scores = cross_val_score(model, x_train, y_train, cv=kfold) ### Takeaway2 ###
    print('{0} scores : {1}'.format(str(model), scores))

# KNeighborsClassifier() scores :   [0.92 1.   0.96 1.   0.96]
# DecisionTreeClassifier() scores : [0.96 0.92 0.92 1.   0.88]
# RandomForestClassifier() scores : [0.96 0.96 0.96 1.   0.88]
# LogisticRegression() scores :     [0.96 1.   0.96 1.   0.92]
# LinearSVC() scores :              [1.   0.96 0.96 1.   0.92]
# SVC() scores :                    [0.96 1.   0.96 1.   0.88]