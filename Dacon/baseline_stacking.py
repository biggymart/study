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