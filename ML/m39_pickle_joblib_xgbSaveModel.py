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
model.fit(x_train, y_train, verbose=1, eval_metric=['rmse'], eval_set=[(x_train, y_train), (x_test, y_test)], early_stopping_rounds=10)

#4. score and predict
aaa = model.score(x_test, y_test)
print("model.score : ", aaa)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)

print("=====================")
results = model.evals_result()
# print(results)


### 방법 1 ###
# python에서 제공하는 기능
import pickle
pickle.dump(model, open('../data/xgb_save/m39.pickle.dat', 'wb')) #dump == save, write binary
print('pickle 저장 완료')

model_pic = pickle.load(open('../data/xgb_save/m39.pickle.dat', 'rb'))
print('pickle 불러오기 완료')
r2_pic = model_pic.score(x_test, y_test)
print('r2 pickle :', r2_pic)

### 방법 2 ###
import joblib
joblib.dump(model, '../data/xgb_save/m40.joblib.dat') # pickle과 달리 open 없이 경로만 쓰면 됨
print('joblib 저장하기 완료')

model_job = joblib.load('../data/xgb_save/m40.joblib.dat')
print('joblib 불러오기 완료')
r2_job = model_job.score(x_test, y_test)
print('r2 joblib :', r2_job)

### 방법 3 ###
# xgb 자체
model.save_model("../data/xgb_save/m41.xgb.model")
print('xgb model 저장하기 완료')

model_xgb = XGBRegressor()
model_xgb.load_model('../data/xgb_save/m41.xgb.model')
print('xgb model 불러오기 완료')
r2_xgb = model_xgb.score(x_test, y_test)
print('r2 xgb model : ', r2_xgb)