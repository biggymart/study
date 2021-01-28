from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.utils.testing import all_estimators # 추정치
from sklearn.datasets import load_breast_cancer
import numpy as np
np.set_printoptions(precision=2)
import warnings
warnings.filterwarnings('ignore')

dataset = load_breast_cancer()
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

# AdaBoostClassifier 의 정답률 :                [0.96 0.92 0.96 0.95 1.  ]
# BaggingClassifier 의 정답률 :                 [0.92 0.96 0.95 0.98 0.97]
# BernoulliNB 의 정답률 :                       [0.54 0.64 0.66 0.6  0.66]
# CalibratedClassifierCV 의 정답률 :            [0.88 0.84 0.96 0.93 0.95]
# CategoricalNB 은 없는 놈!
# CheckingClassifier 의 정답률 :                [0. 0. 0. 0. 0.]
# ClassifierChain 은 없는 놈!
# ComplementNB 의 정답률 :                      [0.85 0.84 0.92 0.89 0.9 ]
# DecisionTreeClassifier 의 정답률 :            [0.93 0.92 0.89 0.91 0.89]
# DummyClassifier 의 정답률 :                   [0.47 0.54 0.57 0.58 0.52]
# ExtraTreeClassifier 의 정답률 :               [0.88 0.92 0.89 0.93 0.96]
# ExtraTreesClassifier 의 정답률 :              [0.96 0.97 0.97 0.97 0.98]
# GaussianNB 의 정답률 :                        [0.91 0.92 0.95 0.91 0.98]
# GaussianProcessClassifier 의 정답률 :         [0.87 0.89 0.9  0.93 0.89]
# GradientBoostingClassifier 의 정답률 :        [0.98 0.96 0.96 0.95 1.  ]
# HistGradientBoostingClassifier 의 정답률 :    [0.98 0.93 0.97 0.95 0.99]
# KNeighborsClassifier 의 정답률 :              [0.87 0.89 0.92 0.93 0.92]
# LabelPropagation 의 정답률 :                  [0.47 0.38 0.35 0.43 0.35]
# LabelSpreading 의 정답률 :                    [0.47 0.38 0.35 0.43 0.35]
# LinearDiscriminantAnalysis 의 정답률 :        [0.93 0.91 0.99 0.93 0.96]
# LinearSVC 의 정답률 :                         [0.88 0.62 0.96 0.82 0.66]
# LogisticRegression 의 정답률 :                [0.88 0.91 0.93 0.97 0.97]
# LogisticRegressionCV 의 정답률 :              [0.92 0.93 0.95 0.98 0.98]
# MLPClassifier 의 정답률 :                     [0.87 0.89 0.93 0.95 0.96]
# MultiOutputClassifier 은 없는 놈!
# MultinomialNB 의 정답률 :                     [0.85 0.84 0.92 0.9  0.9 ]
# NearestCentroid 의 정답률 :                   [0.82 0.87 0.93 0.89 0.9 ]
# NuSVC 의 정답률 :                             [0.81 0.84 0.93 0.88 0.89]
# OneVsOneClassifier 은 없는 놈!
# OneVsRestClassifier 은 없는 놈!
# OutputCodeClassifier 은 없는 놈!
# PassiveAggressiveClassifier 의 정답률 :       [0.87 0.86 0.68 0.68 0.85]
# Perceptron 의 정답률 :                        [0.87 0.86 0.78 0.91 0.82]
# QuadraticDiscriminantAnalysis 의 정답률 :     [0.91 0.93 0.97 0.96 0.97]
# RadiusNeighborsClassifier 은 없는 놈!
# RandomForestClassifier 의 정답률 :            [0.95 0.96 0.96 0.97 0.97]
# RidgeClassifier 의 정답률 :                   [0.95 0.9  0.97 0.96 0.99]
# RidgeClassifierCV 의 정답률 :                 [0.95 0.9  0.98 0.95 1.  ]
# SGDClassifier 의 정답률 :                     [0.89 0.88 0.71 0.91 0.95]
# SVC 의 정답률 :                               [0.87 0.87 0.95 0.92 0.92]
# StackingClassifier 은 없는 놈!
# VotingClassifier 은 없는 놈!
