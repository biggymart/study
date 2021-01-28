import warnings
warnings.filterwarnings('ignore')

#1. data
import numpy as np
from sklearn.datasets import load_boston
dataset = load_boston()
x = dataset.data
y = dataset.target

#1-0. preprocessing
seed = 66
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV ### Takeaway1 ###
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = seed)
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

# dictionary within list
parameters = [{"n_estimators" : [100, 200], "max_depth" : [6, 8, 10, 12], "min_samples_leaf" : [3, 5, 7, 10], "min_samples_split" : [2, 3, 5, 10], "n_jobs" : [-1, 2, 4]}] ### Takeaway2 ###

#2. model
from sklearn.ensemble import RandomForestRegressor
model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv=kfold) ### Takeaway3 ###

#3. fit
model.fit(x_train, y_train)

#4. score
np.set_printoptions(precision=2)
print('최적의 매개변수: {0}'.format(model.best_estimator_)) ### Takeaway4 ###

from sklearn.metrics import r2_score
y_pred = model.predict(x_test)
print('최종정답률: {0}'.format(r2_score(y_test, y_pred))) ### Takeaway5 ###
# Alternative: model.score(x_test, y_test) 

# 최적의 매개변수: RandomForestRegressor(max_depth=10, min_samples_leaf=3, min_samples_split=3, n_jobs=2)
# 최종정답률: 0.9218811092037417