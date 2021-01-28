# kfold + estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.utils.testing import all_estimators # 추정치
from sklearn.datasets import load_wine
import numpy as np
np.set_printoptions(precision=2)
import warnings
warnings.filterwarnings('ignore')

dataset = load_wine()
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

# AdaBoostClassifier 의 정답률 :                [0.93 0.34 0.89 0.79 0.96]
# BaggingClassifier 의 정답률 :                 [0.97 0.9  0.93 1.   1.  ]
# BernoulliNB 의 정답률 :                       [0.38 0.31 0.43 0.36 0.5 ]
# CalibratedClassifierCV 의 정답률 :            [0.9  0.86 0.89 0.96 0.96]
# CategoricalNB 은 없는 놈!
# CheckingClassifier 의 정답률 :                [0. 0. 0. 0. 0.]
# ClassifierChain 은 없는 놈!
# ComplementNB 의 정답률 :                      [0.83 0.52 0.64 0.57 0.75]
# DecisionTreeClassifier 의 정답률 :            [0.93 0.9  0.93 0.86 0.93]
# DummyClassifier 의 정답률 :                   [0.31 0.28 0.36 0.43 0.29]
# ExtraTreeClassifier 의 정답률 :               [0.97 0.76 0.82 0.96 0.82]
# ExtraTreesClassifier 의 정답률 :              [1. 1. 1. 1. 1.]
# GaussianNB 의 정답률 :                        [0.93 0.97 1.   1.   1.  ]
# GaussianProcessClassifier 의 정답률 :         [0.34 0.52 0.54 0.64 0.39]
# GradientBoostingClassifier 의 정답률 :        [0.86 0.97 0.96 0.96 0.96]
# HistGradientBoostingClassifier 의 정답률 :    [0.97 0.97 1.   1.   1.  ]
# KNeighborsClassifier 의 정답률 :              [0.69 0.72 0.75 0.68 0.75]
# LabelPropagation 의 정답률 :                  [0.52 0.41 0.39 0.39 0.43]
# LabelSpreading 의 정답률 :                    [0.52 0.41 0.39 0.39 0.43]
# LinearDiscriminantAnalysis 의 정답률 :        [1.   0.97 0.96 1.   0.96]
# LinearSVC 의 정답률 :                         [0.86 0.72 0.89 0.82 0.96]
# LogisticRegression 의 정답률 :                [0.93 0.9  0.93 1.   0.96]
# LogisticRegressionCV 의 정답률 :              [0.93 0.86 0.93 1.   1.  ]
# MLPClassifier 의 정답률 :                     [0.93 0.83 0.21 0.36 0.21]  
# MultiOutputClassifier 은 없는 놈!
# MultinomialNB 의 정답률 :                     [0.86 0.83 0.82 0.86 0.86]  
# NearestCentroid 의 정답률 :                   [0.72 0.66 0.79 0.82 0.71]
# NuSVC 의 정답률 :                             [0.97 0.76 0.82 0.96 0.93]
# OneVsOneClassifier 은 없는 놈!
# OneVsRestClassifier 은 없는 놈!
# OutputCodeClassifier 은 없는 놈!
# PassiveAggressiveClassifier 의 정답률 :       [0.1  0.48 0.64 0.64 0.46]
# Perceptron 의 정답률 :                        [0.55 0.62 0.71 0.61 0.61]
# QuadraticDiscriminantAnalysis 의 정답률 :     [0.93 1.   1.   0.93 0.96]
# RadiusNeighborsClassifier 은 없는 놈!
# RandomForestClassifier 의 정답률 :            [1.   0.97 1.   1.   1.  ]
# RidgeClassifier 의 정답률 :                   [1.   0.97 0.96 1.   1.  ]       
# RidgeClassifierCV 의 정답률 :                 [1.   0.97 0.96 1.   1.  ]
# SGDClassifier 의 정답률 :                     [0.21 0.45 0.54 0.64 0.68]
# SVC 의 정답률 :                               [0.72 0.66 0.71 0.75 0.75]
# StackingClassifier 은 없는 놈!
# VotingClassifier 은 없는 놈!