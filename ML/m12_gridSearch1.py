# 선별적으로 모델을 선택하고 파라미터 튜닝까지 함
# 촘촘한 망처럼 다 싸그리 낚아올리겠다 "Grid: 격자 --> 그물망"
# One of generic method for parameter searching
# https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/
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
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV ### Takeaway1 ###
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = seed)
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

# dictionary within list
parameters = [
    {"C" : [1, 10, 100, 1000], "kernel" : ["linear"]}, 
    {"C" : [1, 10, 100], "kernel" : ["rbf"], "gamma" : [0.001, 0.0001]}, 
    {"C" : [1, 10, 100, 1000], "kernel" : ["sigmoid"], "gamma" : [0.001, 0.0001]}
] ### Takeaway2 ###
# each corresponds to nodes, activation, and learning rate

#2. model
from sklearn.svm import SVC
model = GridSearchCV(SVC(), parameters, cv=kfold) ### Takeaway3 ###
# Total running cycles: (4 + 6 + 8) * 5 = 90

#3. fit
model.fit(x_train, y_train)

#4. score
np.set_printoptions(precision=2)
print('최적의 매개변수: {0}'.format(model.best_estimator_)) ### Takeaway4 ###

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print('최종정답률: {0}'.format(accuracy_score(y_test, y_pred))) ### Takeaway5 ###
# Alternative: model.score(x_test, y_test) 

# 최적의 매개변수: SVC(C=1, kernel='linear')
# 최종정답률: 0.9666666666666667
