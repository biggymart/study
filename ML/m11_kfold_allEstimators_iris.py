# kfold + all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.utils.testing import all_estimators # 추정치
from sklearn.datasets import load_iris
import numpy as np
np.set_printoptions(precision=2) # my personal code, render a short number for prints
import warnings
warnings.filterwarnings('ignore')

dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 44)

kfold = KFold(n_splits=5, shuffle=True, random_state=66)

allAlgorithms = all_estimators(type_filter='classifier')
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


# AdaBoostClassifier 의 정답률 :                [0.92 0.96 0.88 0.96 0.96]
# BaggingClassifier 의 정답률 :                 [0.96 0.96 0.88 0.96 0.92]
# BernoulliNB 의 정답률 :                       [0.29 0.25 0.33 0.29 0.17]
# CalibratedClassifierCV 의 정답률 :            [0.79 0.92 0.83 0.96 0.83]
# CategoricalNB 의 정답률 :                     [0.92 0.96 0.88 0.96 0.96]
# CheckingClassifier 의 정답률 :                [0. 0. 0. 0. 0.]
# ClassifierChain 은 없는 놈!
# ComplementNB 의 정답률 :                      [0.62 0.75 0.67 0.71 0.54]
# DecisionTreeClassifier 의 정답률 :            [0.96 0.96 0.88 0.92 0.96]
# DummyClassifier 의 정답률 :                   [0.33 0.25 0.29 0.25 0.38]
# ExtraTreeClassifier 의 정답률 :               [0.96 1.   0.88 0.92 0.96]
# ExtraTreesClassifier 의 정답률 :              [1.   1.   0.88 0.96 0.92]
# GaussianNB 의 정답률 :                        [1.   1.   0.83 1.   0.96]
# GaussianProcessClassifier 의 정답률 :         [0.96 1.   0.92 0.96 0.96]
# GradientBoostingClassifier 의 정답률 :        [0.96 0.96 0.88 0.96 0.96]
# HistGradientBoostingClassifier 의 정답률 :    [1.   0.96 0.88 0.96 0.96]
# KNeighborsClassifier 의 정답률 :              [0.96 1.   0.88 0.96 0.96]
# LabelPropagation 의 정답률 :                  [1.   1.   0.92 0.96 0.96]
# LabelSpreading 의 정답률 :                    [1.   1.   0.92 0.96 0.96]
# LinearDiscriminantAnalysis 의 정답률 :        [0.96 1.   0.96 1.   0.92]
# LinearSVC 의 정답률 :                         [1.   0.96 0.88 0.96 0.92]
# LogisticRegression 의 정답률 :                [0.96 1.   0.92 0.96 0.96]
# LogisticRegressionCV 의 정답률 :              [0.96 1.   0.88 1.   0.92]
# MLPClassifier 의 정답률 :                     [0.96 1.   0.96 0.96 0.92]
# MultiOutputClassifier 은 없는 놈!
# MultinomialNB 의 정답률 :                     [0.83 0.58 0.92 0.75 0.67]
# NearestCentroid 의 정답률 :                   [0.96 1.   0.83 0.92 0.92]
# NuSVC 의 정답률 :                             [0.96 0.96 0.92 0.96 0.96]
# OneVsOneClassifier 은 없는 놈!
# OneVsRestClassifier 은 없는 놈!
# OutputCodeClassifier 은 없는 놈!
# PassiveAggressiveClassifier 의 정답률 :       [0.92 0.58 0.88 0.88 0.88]
# Perceptron 의 정답률 :                        [0.88 0.75 0.92 0.71 0.54]
# QuadraticDiscriminantAnalysis 의 정답률 :     [0.92 1.   1.   1.   0.96]
# RadiusNeighborsClassifier 의 정답률 :         [0.96 0.96 0.88 0.96 0.96]
# RandomForestClassifier 의 정답률 :            [1.   1.   0.88 0.96 0.92]
# RidgeClassifier 의 정답률 :                   [0.83 0.88 0.75 1.   0.71]
# RidgeClassifierCV 의 정답률 :                 [0.83 0.88 0.75 1.   0.71]
# SGDClassifier 의 정답률 :                     [0.67 0.54 0.75 0.71 0.54]
# SVC 의 정답률 :                               [0.96 0.96 0.92 0.96 0.96]
# StackingClassifier 은 없는 놈!
# VotingClassifier 은 없는 놈!