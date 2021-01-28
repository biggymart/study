# kfold + estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.utils.testing import all_estimators # 추정치
from sklearn.datasets import load_boston
import numpy as np
np.set_printoptions(precision=2)
import warnings
warnings.filterwarnings('ignore')

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 44)

kfold = KFold(n_splits=5, shuffle=True, random_state=66)

allAlgorithms = all_estimators(type_filter='regressor')
for (name, algorithm) in allAlgorithms: # 인자 두 개를 가짐
    try:
        model = algorithm()

        scores = cross_val_score(model, x_train, y_train, cv=kfold) # cv=5 가능하지만 셔플이 안 됨
        print(name, '의 정답률 : ', scores)
    except:
        # continue
        print(name, '은 없는 놈!')

import sklearn
print(sklearn.__version__) # 0.23.2 

# ARDRegression 의 정답률 :                     [0.74 0.65 0.69 0.65 0.66]
# AdaBoostRegressor 의 정답률 :                 [0.78 0.78 0.74 0.78 0.82]
# BaggingRegressor 의 정답률 :                  [0.8  0.74 0.83 0.86 0.86]
# BayesianRidge 의 정답률 :                     [0.71 0.69 0.69 0.64 0.67]   
# CCA 의 정답률 :                               [0.73 0.63 0.68 0.56 0.69]
# DecisionTreeRegressor 의 정답률 :             [0.76 0.37 0.68 0.8  0.76]
# DummyRegressor 의 정답률 :                    [-0.   -0.01 -0.02 -0.28 -0.07]  
# ElasticNet 의 정답률 :                        [0.66 0.65 0.67 0.64 0.61]
# ElasticNetCV 의 정답률 :                      [0.64 0.63 0.67 0.64 0.59]      
# ExtraTreeRegressor 의 정답률 :                [0.68 0.6  0.64 0.68 0.8 ]
# ExtraTreesRegressor 의 정답률 :               [0.87 0.81 0.89 0.85 0.87]
# GammaRegressor 의 정답률 :                    [-0.   -0.01 -0.03 -0.2  -0.08]
# GaussianProcessRegressor 의 정답률 :          [-5.5  -7.26 -5.91 -7.96 -5.82]
# GeneralizedLinearRegressor 의 정답률 :        [0.62 0.64 0.66 0.62 0.59]
# GradientBoostingRegressor 의 정답률 :         [0.89 0.84 0.83 0.84 0.91]
# HistGradientBoostingRegressor 의 정답률 :     [0.85 0.75 0.85 0.82 0.84]
# HuberRegressor 의 정답률 :                    [0.51 0.53 0.61 0.56 0.62]
# IsotonicRegression 의 정답률 :                [nan nan nan nan nan]
# KNeighborsRegressor 의 정답률 :               [0.28 0.39 0.51 0.49 0.46]
# KernelRidge 의 정답률 :                       [0.66 0.62 0.65 0.65 0.68]
# Lars 의 정답률 :                              [0.73 0.68 0.69 0.67 0.68]
# LarsCV 의 정답률 :                            [0.73 0.68 0.7  0.7  0.68]
# Lasso 의 정답률 :                             [0.64 0.64 0.67 0.63 0.58]
# LassoCV 의 정답률 :                           [0.67 0.66 0.68 0.64 0.62]
# LassoLars 의 정답률 :                         [-0.   -0.01 -0.02 -0.28 -0.07]
# LassoLarsCV 의 정답률 :                       [0.73 0.68 0.69 0.69 0.68]
# LassoLarsIC 의 정답률 :                       [0.73 0.63 0.7  0.69 0.68]
# LinearRegression 의 정답률 :                  [0.73 0.69 0.69 0.67 0.68]
# LinearSVR 의 정답률 :                         [ 0.52 -0.08  0.46 -1.7   0.56]
# MLPRegressor 의 정답률 :                      [0.58 0.65 0.63 0.58 0.41]
# MultiOutputRegressor 은 없는 놈!
# MultiTaskElasticNet 의 정답률 :               [nan nan nan nan nan]
# MultiTaskElasticNetCV 의 정답률 :             [nan nan nan nan nan]
# MultiTaskLasso 의 정답률 :                    [nan nan nan nan nan]
# MultiTaskLassoCV 의 정답률 :                  [nan nan nan nan nan]
# NuSVR 의 정답률 :                             [0.1  0.16 0.12 0.37 0.14]
# OrthogonalMatchingPursuit 의 정답률 :         [0.53 0.48 0.57 0.48 0.46]
# OrthogonalMatchingPursuitCV 의 정답률 :       [0.68 0.6  0.67 0.61 0.64]
# PLSCanonical 의 정답률 :                      [-1.42 -3.4  -1.73 -5.3  -1.63]
# PLSRegression 의 정답률 :                     [0.69 0.62 0.65 0.7  0.64]
# PassiveAggressiveRegressor 의 정답률 :        [-0.15  0.07  0.29  0.14  0.14]
# PoissonRegressor 의 정답률 :                  [0.75 0.73 0.77 0.73 0.72]
# RANSACRegressor 의 정답률 :                   [0.63 0.48 0.55 0.74 0.41]
# RadiusNeighborsRegressor 은 없는 놈!
# RandomForestRegressor 의 정답률 :             [0.83 0.76 0.81 0.87 0.89]
# RegressorChain 은 없는 놈!
# Ridge 의 정답률 :                             [0.72 0.69 0.7  0.66 0.68]
# RidgeCV 의 정답률 :                           [0.73 0.69 0.7  0.67 0.68]
# SGDRegressor 의 정답률 :                      [-2.69e+25 -4.27e+25 -1.33e+27 -1.39e+26 -2.08e+26]
# SVR 의 정답률 :                               [0.07 0.17 0.06 0.43 0.08]
# StackingRegressor 은 없는 놈!
# TheilSenRegressor 의 정답률 :                 [0.68 0.6  0.6  0.71 0.66]
# TransformedTargetRegressor 의 정답률 :        [0.73 0.69 0.69 0.67 0.68]
# TweedieRegressor 의 정답률 :                  [0.62 0.64 0.66 0.62 0.59]
# VotingRegressor 은 없는 놈!
# _SigmoidCalibration 의 정답률 :               [nan nan nan nan nan]