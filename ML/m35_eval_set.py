# eval_set, evals_result
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import numpy as np


#1. data
# x, y = load_boston(return_X_y=True)
# dataset = load_boston()
# dataset = load_wine()
dataset = load_breast_cancer()
x = dataset.data
y = dataset['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)

#2. model
# model = XGBRegressor(n_estimators=3, learning_rate=0.01, n_jobs=8, use_label_encoder=False) # n_estimators는 epochs와 비슷
model = XGBClassifier(n_estimators=3, learning_rate=0.01, n_jobs=8, use_label_encoder=False)

#3. fit
# model.fit(x_train, y_train, verbose=1, eval_metric='rmse', eval_set=[(x_train, y_train), (x_test, y_test)]) # regression
# model.fit(x_train, y_train, verbose=1, eval_metric='mlogloss', eval_set=[(x_train, y_train), (x_test, y_test)]) # multi classification
model.fit(x_train, y_train, verbose=1, eval_metric='logloss', eval_set=[(x_train, y_train), (x_test, y_test)]) # binary classification

#4. score and predict
aaa = model.score(x_test, y_test) # 반환되는 값은 r2, accuracy
print("aaa : ", aaa)

y_pred = model.predict(x_test)
# r2 = r2_score(y_test, y_pred)
# print("r2 : ", r2)
acc = accuracy_score(y_test, y_pred)
print("acc : ", acc)

print("=====================")
results = model.evals_result()
print(results)

# boston
# [0]     validation_0-rmse:23.61117      validation_1-rmse:23.77772
# [1]     validation_0-rmse:23.38760      validation_1-rmse:23.54969
# [2]     validation_0-rmse:23.16623      validation_1-rmse:23.32398
# aaa :  -5.508600212024355
# r2 :  -5.508600212024355
# =====================
# {'validation_0': OrderedDict([('rmse', [23.611168, 23.387598, 23.166225])]), 'validation_1': OrderedDict([('rmse', [23.777716, 23.54969, 23.323978])])}

# wine
# [0]     validation_0-mlogloss:1.08551   validation_1-mlogloss:1.08594
# [1]     validation_0-mlogloss:1.07263   validation_1-mlogloss:1.07381
# [2]     validation_0-mlogloss:1.05996   validation_1-mlogloss:1.06192
# aaa :  1.0
# acc :  1.0
# =====================
# {'validation_0': OrderedDict([('mlogloss', [1.085513, 1.072629, 1.059957])]), 'validation_1': OrderedDict([('mlogloss', [1.085941, 1.073813, 1.061924])])}

# breast_cancer
# [0]     validation_0-logloss:0.68421    validation_1-logloss:0.68477
# [1]     validation_0-logloss:0.67550    validation_1-logloss:0.67663
# [2]     validation_0-logloss:0.66696    validation_1-logloss:0.66858
# aaa :  0.9649122807017544
# acc :  0.9649122807017544
# =====================
# {'validation_0': OrderedDict([('logloss', [0.684206, 0.675503, 0.666963])]), 'validation_1': OrderedDict([('logloss', [0.684766, 0.676633, 0.668579])])}