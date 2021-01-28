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

# ARDRegression 의 정답률 :                     0.7512651671065581
# AdaBoostRegressor 의 정답률 :                 0.8387724121644411
# BaggingRegressor 의 정답률 :                  0.8618553493285555
# BayesianRidge 의 정답률 :                     0.7444785336818114
# CCA 의 정답률 :                               0.7270542664211517
# DecisionTreeRegressor 의 정답률 :             0.8209778516805106
# DummyRegressor 의 정답률 :                    -0.0007982049217318821
# ElasticNet 의 정답률 :                        0.6990500898755508
# ElasticNetCV 의 정답률 :                      0.6902681369495264
# ExtraTreeRegressor 의 정답률 :                0.7984828552037211
# ExtraTreesRegressor 의 정답률 :               0.9011051150206106
# GammaRegressor 의 정답률 :                    -0.0007982049217318821
# GaussianProcessRegressor 의 정답률 :          -5.639147690233129
# GeneralizedLinearRegressor 의 정답률 :        0.6917874063129013
# GradientBoostingRegressor 의 정답률 :         0.8949724490602259
# HistGradientBoostingRegressor 의 정답률 :     0.8991491407747458
# HuberRegressor 의 정답률 :                    0.7233379135400204
# IsotonicRegression 은 없는 놈!
# KNeighborsRegressor 의 정답률 :               0.6390759816821279
# KernelRidge 의 정답률 :                       0.7744886782300767
# Lars 의 정답률 :                              0.7521800808693164
# LarsCV 의 정답률 :                            0.7570138649983484
# Lasso 의 정답률 :                             0.6855879495660049
# LassoCV 의 정답률 :                           0.7154057460487299
# LassoLars 의 정답률 :                         -0.0007982049217318821
# LassoLarsCV 의 정답률 :                       0.7570138649983484
# LassoLarsIC 의 정답률 :                       0.754094595988446
# LinearRegression 의 정답률 :                  0.7521800808693141
# LinearSVR 의 정답률 :                         0.30765194516863636
# MLPRegressor 의 정답률 :                      0.5216266091374727
# MultiOutputRegressor 은 없는 놈!
# MultiTaskElasticNet 은 없는 놈!
# MultiTaskElasticNetCV 은 없는 놈!
# MultiTaskLasso 은 없는 놈!
# MultiTaskLassoCV 은 없는 놈!
# NuSVR 의 정답률 :                             0.32534704254368274
# OrthogonalMatchingPursuit 의 정답률 :         0.5661769106723642
# OrthogonalMatchingPursuitCV 의 정답률 :       0.7377665753906506
# PLSCanonical 의 정답률 :                      -1.7155095545127699
# PLSRegression 의 정답률 :                     0.7666940310402938
# PassiveAggressiveRegressor 의 정답률 :        -1.0063351806394718
# PoissonRegressor 의 정답률 :                  0.8014250117852569
# RANSACRegressor 의 정답률 :                   0.4265949667045228
# RadiusNeighborsRegressor 은 없는 놈!
# RandomForestRegressor 의 정답률 :             0.8923473684706418
# RegressorChain 은 없는 놈!
# Ridge 의 정답률 :                             0.7539303499010775
# RidgeCV 의 정답률 :                           0.7530092298810112
# SGDRegressor 의 정답률 :                      -8.680593662892233e+25
# SVR 의 정답률 :                               0.2868662719877668
# StackingRegressor 은 없는 놈!
# TheilSenRegressor 의 정답률 :                 0.7904146806427417
# TransformedTargetRegressor 의 정답률 :        0.7521800808693141
# TweedieRegressor 의 정답률 :                  0.6917874063129013
# VotingRegressor 은 없는 놈!
# _SigmoidCalibration 은 없는 놈!