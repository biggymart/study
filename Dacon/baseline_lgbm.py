import time
import random
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
filepath = 'C:/data/mnist/'
train = pd.read_csv(filepath + 'train.csv') # (2048, 787)
test  = pd.read_csv(filepath + 'test.csv') # (20480, 786)
submission = pd.read_csv(filepath +'submission.csv')

from scipy.signal import correlate2d


# 문자 데이터를 one-hot encoding하고
# 이미지 픽셀 데이터를 784개의 위치 feature라고 생각하고 concat
X_train = pd.concat(
    (pd.get_dummies(train.letter), train[[str(i) for i in range(784)]]), 
    axis=1)
y_train = train['digit']

# Train set을 8:2로 분리
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

from lightgbm import LGBMClassifier

lgb = LGBMClassifier()

# 모델 적합
lgb.fit(X_train, y_train)

# 예측 정확도 출력
print((lgb.predict(X_valid) == y_valid.values).sum() / len(y_valid))

# Test 데이터에 대해 예측을 진행
X_test = pd.concat(
    (pd.get_dummies(test.letter), test[[str(i) for i in range(784)]]), 
axis=1)

# Submission 컬럼에 이를 기록
submission.digit = lgb.predict(X_test)

# 파일로 저장 후 업로드
submission.to_csv('first_submission.csv', index=False) # 57.84313725% 의 결과를 얻음