from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils.testing import all_estimators # 추정치
from sklearn.datasets import load_boston
import warnings

warnings.filterwarnings('ignore')

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 44)

allAlgorithms = all_estimators(type_filter='regressor') ### Takeaway1 ###

for (name, algorithm) in allAlgorithms: # 인자 두 개를 가짐
    try: # 예외처리
        model = algorithm()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', r2_score(y_test, y_pred)) ### Takeaway2 ###
    except:
        # continue
        print(name, '은 없는 놈!')

import sklearn
print(sklearn.__version__) # 0.23.2 

# 이 버전에 있는 모든 regressor 나타내줌
