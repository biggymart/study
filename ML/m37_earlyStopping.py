# eval_set, evals_result
from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np


#1. data
dataset = load_boston()
x = dataset.data
y = dataset['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)

#2. model
model = XGBRegressor(n_estimators=1000, learning_rate=0.01, n_jobs=8, use_label_encoder=False)

#3. fit
model.fit(x_train, y_train, verbose=1, eval_metric=['rmse'], eval_set=[(x_train, y_train), (x_test, y_test)], early_stopping_rounds=10) # 이 한 부분만 추가하면 된다

#4. score and predict
aaa = model.score(x_test, y_test)
print("aaa : ", aaa)
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)

print("=====================")
results = model.evals_result()
print(results)

# boston
# ...
# [505]   validation_0-rmse:0.93773       validation_1-rmse:2.36673
# aaa :  0.93302293398985
# r2 :  0.93302293398985
# =====================
# {'validation_0': OrderedDict([('rmse', [23.611168, ...)])}