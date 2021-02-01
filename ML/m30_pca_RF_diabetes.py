# PCA의 적용; model = RandomForest
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import r2_score


#1. data and preprocessing
datasets = load_diabetes()
x = datasets.data
y = datasets.target

pca = PCA(n_components=8) # evr = 0.95
x_pca = pca.fit_transform(x)

seed = 66
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, train_size=0.8, shuffle=True, random_state=seed)
kfold = KFold(n_splits=5, shuffle=True, random_state=seed) 

#2. model
parameters = [{"aaa__n_estimators" : [100, 200], "aaa__max_depth" : [6, 8, 10, 12], "aaa__min_samples_leaf" : [3, 5, 7, 10], "aaa__min_samples_split" : [2, 3, 5, 10], "aaa__n_jobs" : [-1, 2, 4]}]

#3. fit
scaler = MinMaxScaler() # or StandardScaler
pipe = Pipeline([("scaler", scaler), ('aaa', RandomForestRegressor())])
model = RandomizedSearchCV(pipe, parameters, cv=kfold)

#4. score and predict
score = cross_val_score(model, x_train, y_train, cv=kfold) # nested cv
print(score)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('최종정답률: {0}'.format(r2_score(y_test, y_pred)))

# [0.49655907 0.50385721 0.42425402 0.52801113 0.47071933]
# 최종정답률: 0.38671721429865014
