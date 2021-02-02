'''
https://dacon.io/competitions/official/235626/codeshare/1668
https://dacon.io/competitions/official/235626/codeshare/1624
https://dacon.io/competitions/official/235626/codeshare/1551

sklearn으로 이미지 분류하는 것 쉽게 알려주는
https://kapernikov.com/tutorial-image-classification-with-scikit-learn/
문제의 성격: 다중분류
특징: 
(1) 의도적인 노이즈 첨가됨
(2) train의 양이 적음
요구사항: ML로 구성해볼 것

cv score; searchcv; pipeline; xgb; pca; early stopping; eval metrics; model save 
'''
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#1. data
# load data from csv
filepath = 'C:/data/mnist/'
train = pd.read_csv(filepath + 'train.csv') # (2048, 787)
test  = pd.read_csv(filepath + 'test.csv') # (20480, 786)
submission = pd.read_csv(filepath + 'submission.csv') # (20480, 2)

# slicing
train_copy = train.copy()
x_train_pd = train_copy.loc[:, 'letter':'783'] 
y_train_pd = train_copy['digit'] 
x = x_train_pd.to_numpy() # (2048, 785)
y = y_train_pd.to_numpy() # (2048,)

test_copy = test.copy() # (20480, 786)
x_test_pd = test_copy.loc[:, 'letter':'783']
x_test = x_test_pd.to_numpy() # (20480, 785)

# convert string to float
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 0] = le.fit_transform(x[:, 0]) # x.shape (2048, 785)
x_test[:, 0] = le.fit_transform(x_test[:, 0]) # (20480, 785)


# train_test_split
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
seed = 66
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=seed)
# kfold = KFold(n_splits=5, shuffle=True, random_state=seed) 

from xgboost import XGBClassifier
model = XGBClassifier(learning_rate=0.01, n_jobs=8, use_label_encoder=False) # n_estimators

#2, 3. model and fit # Optional (simple version)
# model.fit(x_train, y_train, verbose=1, eval_metric=['mlogloss'], eval_set=[(x_train, y_train), (x_test, y_test)], early_stopping_rounds=10)

# dictionary within list
# parameters = [{"n_estimators" : [100, 200], "max_depth" : [8, 10, 12]}] 

#2. model
# model = RandomizedSearchCV(model, parameters, cv=kfold)

#3. fit
model.fit(x_train, y_train, verbose=1, eval_metric=['mlogloss', 'merror'], eval_set=[(x_train, y_train), (x_val, y_val)], early_stopping_rounds=10)


#4. score
# print('최적의 매개변수: {0}'.format(model.best_estimator_))

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_val)
print('최종정답률: {0}'.format(accuracy_score(y_val, y_pred)))


# Submission 컬럼에 이를 기록
submission.digit = model.predict(x_test)

# 파일로 저장 후 업로드
submission.to_csv('first_submission.csv', index=False) # 0.3


'''
# #3. fit
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.pipeline import Pipeline, make_pipeline
# scaler = MinMaxScaler() # or StandardScaler()
# pipe = Pipeline([("scaler", scaler), ('aaa', xgb)])
# model = RandomizedSearchCV(pipe, parameters, cv=kfold)

#4. score and predict
model.fit(x_train, y_train)
score = model.score(x_test, y_test) # nested cv
print(score)


############ BASELINE ##############
# train_test_split -> lgbm model -> fit 끝


# Test 데이터에 대해 예측을 진행
X_test = pd.concat(
    (pd.get_dummies(test.letter), test[[str(i) for i in range(784)]]), 
axis=1)

# Submission 컬럼에 이를 기록
submission.digit = lgb.predict(X_test)

# 파일로 저장 후 업로드
submission.to_csv('first_submission.csv', index=False) # 57.84313725% 의 결과를 얻음
'''

# 최적의 매개변수: XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#               importance_type='gain', interaction_constraints='',
#               learning_rate=0.01, max_delta_step=0, max_depth=8,
#               min_child_weight=1, missing=nan, monotone_constraints='()',
#               n_estimators=100, n_jobs=8, num_parallel_tree=1,
#               objective='multi:softprob', random_state=0, reg_alpha=0,
#               reg_lambda=1, scale_pos_weight=None, subsample=1,
#               tree_method='exact', use_label_encoder=False,
#               validate_parameters=1, verbosity=None)
# 최종정답률: 0.3926829268292683

'''
feature_names = list(test)
feature_names.remove('letter')

train[feature_names] = (train[feature_names] / 255)
test[feature_names] = (test[feature_names] / 255)

train= pd.get_dummies(train, columns=['letter'])
test = pd.get_dummies(test, columns=['letter'])

Xtrain = train[list(test)]
Xtest = test.copy()
Ytrain = np.array(train['digit'])




# Final 16th/Public:0.921569/Stacking Random Model

# 랜덤한 하이퍼파라미터로 만든 약 400여 개의 모델을 스태킹했습니다.  

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = './data'
from tqdm import tqdm

from scipy.stats import loguniform

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import ExtraTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB

import keras
import keras.backend as K
from keras.layers import (Dense, Input, Activation, Convolution2D, MaxPooling2D, BatchNormalization, 
                          Dropout, Flatten, SpatialDropout2D, Add, Concatenate , Reshape,
                         GlobalAveragePooling2D)
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import Nadam

def mish(x):
    return x * K.tanh(K.softplus(x))

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score

import optuna
import random

# Lv. 0 에서는 케라스로 만든 24개의 모델과, 다른 알고리즘들을 랜덤한 파라미터로 학습시켜 총 428개의 모델을 만듭니다.  
# 케라스 모델은 나름 열심히 실험해서 만들었지만 정확도를 높이는 것에 한계를 느껴서, 나머지 모델들의 집단지성으로 케라스 모델이 틀렸던 부분 몇 개를 더 맞추게 하는 것이 목표입니다. 

num_models=50
lgb_params_list=[]

for _ in range(num_models):
    params = {'n_estimators' : np.random.randint(low=100, high=1000), 
             'colsample_bytree' : np.random.uniform(low=0.5, high=1), 
             'subsample' : np.random.uniform(low=0.5, high=1), 
             'reg_alpha' : np.random.uniform(low=0, high=30), 
             'reg_lambda' : np.random.uniform(low=0, high=30),
             'learning_rate' : np.random.uniform(low=0.01, high=0.3), 
             'drop_rate' : np.random.uniform(low=0.3, high=0.8),
             'uniform_drop' : np.random.choice([True, False]),
              'num_leaves' : np.random.randint(low=7, high=128)
             }
    lgb_params_list.append(params)
    
    model =  LGBMClassifier(boosting_type='dart', objective='softmax', tree_learner='feature', num_class=10, subsample_freq=1, 
                       random_state=18, max_drop=-1,
                      **params)
                           
    train_preds = cross_val_predict(model, Xtrain, Ytrain, cv=5, method='predict_proba')
    print(f"Score : {accuracy_score(Ytrain, np.argmax(train_preds, axis=1))}")
    stack_train_1 = np.concatenate((stack_train_1, train_preds), axis=1)
    model.fit(Xtrain, Ytrain)
    stack_test_1 = np.concatenate((stack_test_1, model.predict_proba(Xtest)), axis=1)

    pd.DataFrame(stack_train_1).to_csv('stack_train_1.csv', index=False)
    pd.DataFrame(stack_test_1).to_csv('stack_test_1.csv', index=False)

    print(stack_train_1.shape, stack_test_1.shape, '\n')

lgb_params_list = pd.DataFrame.from_dict(lgb_params_list)
lgb_params_list.to_csv('lgb_params_list.csv', index=False)
lgb_params_list.head(10)

XGboost
params = {'n_estimators' : np.random.randint(low=100, high=1000), 
             'colsample_bytree' : np.random.uniform(low=0.5, high=1), 
             'subsample' : np.random.uniform(low=0.5, high=1), 
             'reg_alpha' : np.random.uniform(low=0, high=30), 
             'reg_lambda' : np.random.uniform(low=0, high=30),
             'learning_rate' : np.random.uniform(low=0.01, high=0.3),
             'max_depth' : np.random.randint(low=3, high=8)
             }
             '''