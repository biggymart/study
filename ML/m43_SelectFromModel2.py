# 실습
# 1. 상단 모델에 그리드서치 또는 랜덤서치로 튜닝한 모델 구성
# 최적의 R2값과 피처임포턴스 구할 것

# 2. 위 쓰레드 값으로 SelectFromModel을 구해서
# 최적의 피처 개수를 구할 것

# 3. 위 피처 개수로 데이터(피처)를 수정(삭제)해서
# 그리드서치 또는 랜덤서치 적용하여 
# 최적의 R2 구할 것

# 1번 값과 2번 값 비교

import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score

# feature importance 친구
x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=66
)

model = XGBRegressor(n_jobs=8)

parameters = [
    {"n_estimators" : [100, 200, 300], "learning_rate" : [0.001, 0.01, 0.1, 0.3], "max_depth" : [4, 5, 6]},
    {"n_estimators" : [90, 100, 110], "learning_rate" : [0.001, 0.01, 0.1], "max_depth" : [4, 5, 6], "colsample_bytree" : [0.6, 0.9, 1]},
    {"n_estimators" : [90, 110], "learning_rate" : [0.001, 0.1, 0.5], "max_depth": [4,5,6], "colsample_bytree" : [0.6, 0.9, 1], "colsample_bylevel" : [0.6, 0.7, 0.9]}
]

kfold = KFold(n_splits=5, shuffle=True, random_state=66)
grid = RandomizedSearchCV(model, parameters, cv=kfold)
grid.fit(x_train, y_train)

score = grid.score(x_test, y_test)
print("R2 :", score)
print("best_params_ :", grid.best_params_)

model2 = grid.best_estimator_
model2.fit(x_train, y_train)

thresholds = np.sort(model2.feature_importances_)
print(thresholds) # np.sum 총합 1

for thresh in thresholds:
    selection = SelectFromModel(model2, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)
    print(select_x_train.shape)

    selection_model = XGBRegressor(n_jobs=8)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1],
            score*100))

    # if문, 변수로 업데이트