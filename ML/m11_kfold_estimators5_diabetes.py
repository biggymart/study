from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.utils.testing import all_estimators # 추정치
from sklearn.datasets import load_diabetes
import numpy as np
np.set_printoptions(precision=2)
import warnings
warnings.filterwarnings('ignore')

dataset = load_diabetes()
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

# ARDRegression 의 정답률 :                     [0.31 0.5  0.47 0.52 0.48]
# AdaBoostRegressor 의 정답률 :                 [0.3  0.39 0.4  0.43 0.46]
# BaggingRegressor 의 정답률 :                  [0.28 0.43 0.3  0.34 0.38]
# BayesianRidge 의 정답률 :                     [0.32 0.51 0.47 0.54 0.47]
# CCA 의 정답률 :                               [0.3  0.48 0.42 0.35 0.47]
# DecisionTreeRegressor 의 정답률 :             [-0.37 -0.34 -0.25 -0.09 -0.08]
# DummyRegressor 의 정답률 :                    [-0.   -0.   -0.01 -0.03 -0.04]
# ElasticNet 의 정답률 :                        [ 0.01  0.01 -0.   -0.02 -0.03]
# ElasticNetCV 의 정답률 :                      [0.38 0.44 0.4  0.47 0.42]
# ExtraTreeRegressor 의 정답률 :                [-0.53 -0.21 -0.07 -0.17  0.08]
# ExtraTreesRegressor 의 정답률 :               [0.27 0.45 0.41 0.43 0.46]
# GammaRegressor 의 정답률 :                    [ 0.    0.   -0.   -0.02 -0.03]
# GaussianProcessRegressor 의 정답률 :          [-12.78 -27.81 -20.19 -23.92 -19.54]
# GeneralizedLinearRegressor 의 정답률 :        [ 0.    0.01 -0.   -0.02 -0.03]
# GradientBoostingRegressor 의 정답률 :         [0.27 0.37 0.45 0.44 0.43]
# HistGradientBoostingRegressor 의 정답률 :     [0.13 0.37 0.43 0.31 0.46]
# HuberRegressor 의 정답률 :                    [0.27 0.52 0.48 0.53 0.47]
# IsotonicRegression 의 정답률 :                [nan nan nan nan nan]
# KNeighborsRegressor 의 정답률 :               [0.15 0.39 0.35 0.42 0.35]
# KernelRidge 의 정답률 :                       [-4.6  -3.13 -3.61 -3.6  -2.84]
# Lars 의 정답률 :                              [ 0.27  0.51 -0.65  0.55  0.03]
# LarsCV 의 정답률 :                            [0.34 0.49 0.44 0.51 0.48]
# Lasso 의 정답률 :                             [0.31 0.35 0.32 0.33 0.35]
# LassoCV 의 정답률 :                           [0.31 0.49 0.44 0.52 0.48]
# LassoLars 의 정답률 :                         [0.34 0.39 0.36 0.37 0.4 ]
# LassoLarsCV 의 정답률 :                       [0.31 0.49 0.44 0.51 0.48]
# LassoLarsIC 의 정답률 :                       [0.34 0.47 0.45 0.51 0.48]
# LinearRegression 의 정답률 :                  [0.27 0.51 0.48 0.55 0.48]
# LinearSVR 의 정답률 :                         [-0.59 -0.37 -0.56 -0.64 -0.2 ]
# MLPRegressor 의 정답률 :                      [-3.6  -2.7  -3.1  -3.09 -1.95]
# MultiOutputRegressor 은 없는 놈!
# MultiTaskElasticNet 의 정답률 :               [nan nan nan nan nan]
# MultiTaskElasticNetCV 의 정답률 :             [nan nan nan nan nan]
# MultiTaskLasso 의 정답률 :                    [nan nan nan nan nan]
# MultiTaskLassoCV 의 정답률 :                  [nan nan nan nan nan]
# NuSVR 의 정답률 :                             [0.14 0.14 0.12 0.11 0.11]
# OrthogonalMatchingPursuit 의 정답률 :         [0.18 0.31 0.28 0.3  0.39]
# OrthogonalMatchingPursuitCV 의 정답률 :       [0.27 0.51 0.41 0.46 0.46]
# PLSCanonical 의 정답률 :                      [-2.07 -1.8  -0.98 -0.81 -1.15]
# PLSRegression 의 정답률 :                     [0.31 0.52 0.47 0.56 0.44]
# PassiveAggressiveRegressor 의 정답률 :        [0.37 0.41 0.41 0.52 0.43]
# PoissonRegressor 의 정답률 :                  [0.31 0.34 0.29 0.35 0.31]
# RANSACRegressor 의 정답률 :                   [-0.08  0.32  0.02  0.34  0.13]
# RadiusNeighborsRegressor 의 정답률 :          [-0.   -0.   -0.01 -0.03 -0.04]
# RandomForestRegressor 의 정답률 :             [0.32 0.41 0.42 0.43 0.47]
# RegressorChain 은 없는 놈!
# Ridge 의 정답률 :                             [0.36 0.41 0.35 0.42 0.38]
# RidgeCV 의 정답률 :                           [0.34 0.5  0.46 0.53 0.47]
# SGDRegressor 의 정답률 :                      [0.35 0.39 0.33 0.42 0.39]
# SVR 의 정답률 :                               [ 0.14  0.15  0.05 -0.    0.15]
# StackingRegressor 은 없는 놈!
# TheilSenRegressor 의 정답률 :                 [0.25 0.51 0.47 0.51 0.49]
# TransformedTargetRegressor 의 정답률 :        [0.27 0.51 0.48 0.55 0.48]
# TweedieRegressor 의 정답률 :                  [ 0.    0.01 -0.   -0.02 -0.03]     
# VotingRegressor 은 없는 놈!
# _SigmoidCalibration 의 정답률 :               [nan nan nan nan nan]
