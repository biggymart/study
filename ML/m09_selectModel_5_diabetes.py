from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils.testing import all_estimators # 추정치
from sklearn.datasets import load_diabetes
import warnings
warnings.filterwarnings('ignore')

dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 44)

allAlgorithms = all_estimators(type_filter='regressor')

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', r2_score(y_test, y_pred))
    except:
        # continue
        print(name, '은 없는 놈!')

import sklearn
print(sklearn.__version__) # 0.23.2 

# 이 버전에 있는 모든 regressor 나타내줌

# ARDRegression 의 정답률 :                     0.5278342233068394
# AdaBoostRegressor 의 정답률 :                 0.4271657172636433
# BaggingRegressor 의 정답률 :                  0.3792142725521154
# BayesianRidge 의 정답률 :                     0.5193410135537663
# CCA 의 정답률 :                               0.48879618038824757
# DecisionTreeRegressor 의 정답률 :             -0.4283392476190997
# DummyRegressor 의 정답률 :                    -0.07457975637038539
# ElasticNet 의 정답률 :                        -0.06518000443720706
# ElasticNetCV 의 정답률 :                      0.4294375480398558
# ExtraTreeRegressor 의 정답률 :                -0.07004648727742224
# ExtraTreesRegressor 의 정답률 :               0.4227674586556637
# GammaRegressor 의 정답률 :                    -0.06869757267027454
# GaussianProcessRegressor 의 정답률 :          -16.57366391984241
# GeneralizedLinearRegressor 의 정답률 :        -0.06771406705799343
# GradientBoostingRegressor 의 정답률 :         0.3634546723687253
# HistGradientBoostingRegressor 의 정답률 :     0.3504135950167052
# HuberRegressor 의 정답률 :                    0.5205018285661304
# IsotonicRegression 은 없는 놈!
# KNeighborsRegressor 의 정답률 :               0.35838503635518537
# KernelRidge 의 정답률 :                       -4.4187445504449405
# Lars 의 정답률 :                              0.21479550446394002
# LarsCV 의 정답률 :                            0.516365352104498
# Lasso 의 정답률 :                             0.33086319953362164
# LassoCV 의 정답률 :                           0.5222186221789182
# LassoLars 의 정답률 :                         0.3570808988866827
# LassoLarsCV 의 정답률 :                       0.5214536844628463
# LassoLarsIC 의 정답률 :                       0.5224736703335271
# LinearRegression 의 정답률 :                  0.525204262124852
# LinearSVR 의 정답률 :                         -0.8306231508702273
# MLPRegressor 의 정답률 :                      -3.979933590212438
# MultiOutputRegressor 은 없는 놈!
# MultiTaskElasticNet 은 없는 놈!
# MultiTaskElasticNetCV 은 없는 놈!
# MultiTaskLasso 은 없는 놈!
# MultiTaskLassoCV 은 없는 놈!
# NuSVR 의 정답률 :                             0.07746639731663862
# OrthogonalMatchingPursuit 의 정답률 :         0.3337053538857254
# OrthogonalMatchingPursuitCV 의 정답률 :       0.5257611661032995
# PLSCanonical 의 정답률 :                      -1.2663831979876923
# PLSRegression 의 정답률 :                     0.5042012880276586
# PassiveAggressiveRegressor 의 정답률 :        0.475471831094896
# PoissonRegressor 의 정답률 :                  0.29880208432725275
# RANSACRegressor 의 정답률 :                   0.1605248095409859
# RadiusNeighborsRegressor 의 정답률 :          -0.07457975637038539
# RandomForestRegressor 의 정답률 :             0.44661787778509787
# RegressorChain 은 없는 놈!
# Ridge 의 정답률 :                             0.40179727975154844
# RidgeCV 의 정답률 :                           0.5132298404989653
# SGDRegressor 의 정답률 :                      0.377425683899379
# SVR 의 정답률 :                               0.008054881772852074
# StackingRegressor 은 없는 놈!
# TheilSenRegressor 의 정답률 :                 0.5236726185038091
# TransformedTargetRegressor 의 정답률 :        0.525204262124852
# TweedieRegressor 의 정답률 :                  -0.06771406705799343
# VotingRegressor 은 없는 놈!
# _SigmoidCalibration 은 없는 놈!
