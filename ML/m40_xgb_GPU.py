from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score

datasets = load_boston()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=66
)

model = XGBRegressor(n_estimators=100000, learning_rate=0.01,
                    tree_method='gpu_hist', ##### GPU로 가동 ########
                    predictor='gpu_predictor', # 자매품: cpu_predictor
                    gpu_id=0 # gpu 선택해서 할 수 있어
) 

model.fit(x_train, y_train, verbose=1, eval_metric=['rmse'],
          eval_set=[(x_train, y_train), (x_test, y_test)],
          early_stopping_rounds=10000 
)

aaa = model.score(x_test, y_test)
print("model.score :", aaa)

