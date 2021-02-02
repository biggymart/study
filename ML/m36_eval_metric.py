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
# model = XGBRegressor(n_estimators=3, learning_rate=0.01, n_jobs=8, use_label_encoder=False) # regression
model = XGBClassifier(n_estimators=3, learning_rate=0.01, n_jobs=8, use_label_encoder=False) # classification

#3. fit
# model.fit(x_train, y_train, verbose=1, eval_metric=['rmse', 'mae'], eval_set=[(x_train, y_train), (x_test, y_test)]) # regression
# model.fit(x_train, y_train, verbose=1, eval_metric=['mlogloss', 'merror'], eval_set=[(x_train, y_train), (x_test, y_test)]) # multi classification
model.fit(x_train, y_train, verbose=1, eval_metric=['logloss', 'error'], eval_set=[(x_train, y_train), (x_test, y_test)]) # binary classification
'''

# eval_metric
eval_metric: 설정 한 objective기본 설정 값이 지정되어 있습니다.
rmse : 제곱 평균 제곱근 오차
mae : 절대 오류를 의미
logloss : 음의 로그 우도
error    : 이진 분류 오류율 (0.5 임계 값)
merror : 다중 클래스 분류 오류율
mlogloss : 다중 클래스 logloss
auc : 곡선 아래 영역 
'''

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
# [0]     validation_0-rmse:23.61117      validation_0-mae:21.77568       validation_1-rmse:23.77772      validation_1-mae:21.98074
# [1]     validation_0-rmse:23.38760      validation_0-mae:21.56418       validation_1-rmse:23.54969      validation_1-mae:21.76486
# [2]     validation_0-rmse:23.16623      validation_0-mae:21.35475       validation_1-rmse:23.32398      validation_1-mae:21.55121
# aaa :  -5.508600212024355
# r2 :  -5.508600212024355
# =====================
# {'validation_0': OrderedDict([('rmse', [23.611168, 23.387598, 23.166225]), ('mae', [21.775681, 21.564184, 21.354753])]), 'validation_1': OrderedDict([('rmse', [23.777716, 23.54969, 23.323978]), ('mae', [21.98074, 21.764856, 21.551214])])}

# wine
# [0]     validation_0-mlogloss:1.08551   validation_0-merror:0.00704     validation_1-mlogloss:1.08594   validation_1-merror:0.02778
# [1]     validation_0-mlogloss:1.07263   validation_0-merror:0.00704     validation_1-mlogloss:1.07381   validation_1-merror:0.00000
# [2]     validation_0-mlogloss:1.05996   validation_0-merror:0.00704     validation_1-mlogloss:1.06192   validation_1-merror:0.00000
# aaa :  1.0
# acc :  1.0
# =====================
# {'validation_0': OrderedDict([('mlogloss', [1.085513, 1.072629, 1.059957]), ('merror', [0.007042, 0.007042, 0.007042])]), 'validation_1': OrderedDict([('mlogloss', [1.085941, 1.073813, 1.061924]), ('merror', [0.027778, 0.0, 0.0])])}

# breast_cancer
# [0]     validation_0-logloss:0.68421    validation_0-error:0.01758      validation_1-logloss:0.68477    validation_1-error:0.03509
# [1]     validation_0-logloss:0.67550    validation_0-error:0.01758      validation_1-logloss:0.67663    validation_1-error:0.03509
# [2]     validation_0-logloss:0.66696    validation_0-error:0.01758      validation_1-logloss:0.66858    validation_1-error:0.03509
# aaa :  0.9649122807017544
# acc :  0.9649122807017544
# =====================
# {'validation_0': OrderedDict([('logloss', [0.684206, 0.675503, 0.666963]), ('error', [0.017582, 0.017582, 0.017582])]), 'validation_1': OrderedDict([('logloss', [0.684766, 0.676633, 0.668579]), ('error', [0.035088, 0.035088, 0.035088])])}