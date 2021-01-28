from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators # 추정치
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings('ignore')

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 44)

allAlgorithms = all_estimators(type_filter='classifier')

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', accuracy_score(y_test, y_pred))
    except:
        # continue
        print(name, '은 없는 놈!')

import sklearn
print(sklearn.__version__) # 0.23.2 

# AdaBoostClassifier 의 정답률 :                0.9736842105263158
# BaggingClassifier 의 정답률 :                 0.956140350877193
# BernoulliNB 의 정답률 :                       0.6578947368421053
# CalibratedClassifierCV 의 정답률 :            0.9824561403508771
# CategoricalNB 은 없는 놈!
# CheckingClassifier 의 정답률 :                0.34210526315789475
# ClassifierChain 은 없는 놈!
# ComplementNB 의 정답률 :                      0.9473684210526315
# DecisionTreeClassifier 의 정답률 :            0.9385964912280702
# DummyClassifier 의 정답률 :                   0.5087719298245614
# ExtraTreeClassifier 의 정답률 :               0.9385964912280702
# ExtraTreesClassifier 의 정답률 :              0.9736842105263158
# GaussianNB 의 정답률 :                        0.9736842105263158
# GaussianProcessClassifier 의 정답률 :         0.9298245614035088
# GradientBoostingClassifier 의 정답률 :        0.9912280701754386
# HistGradientBoostingClassifier 의 정답률 :    0.9736842105263158
# KNeighborsClassifier 의 정답률 :              0.956140350877193
# LabelPropagation 의 정답률 :                  0.3684210526315789
# LabelSpreading 의 정답률 :                    0.3684210526315789
# LinearDiscriminantAnalysis 의 정답률 :        0.9912280701754386
# LinearSVC 의 정답률 :                         0.9736842105263158
# LogisticRegression 의 정답률 :                0.9736842105263158
# LogisticRegressionCV 의 정답률 :              0.9736842105263158
# MLPClassifier 의 정답률 :                     0.9385964912280702
# MultiOutputClassifier 은 없는 놈!
# MultinomialNB 의 정답률 :                     0.9473684210526315
# NearestCentroid 의 정답률 :                   0.9298245614035088
# NuSVC 의 정답률 :                             0.9385964912280702
# OneVsOneClassifier 은 없는 놈!
# OneVsRestClassifier 은 없는 놈!
# OutputCodeClassifier 은 없는 놈!
# PassiveAggressiveClassifier 의 정답률 :       0.9385964912280702
# Perceptron 의 정답률 :  0.8421052631578947
# QuadraticDiscriminantAnalysis 의 정답률 :     0.9649122807017544
# RadiusNeighborsClassifier 은 없는 놈!
# RandomForestClassifier 의 정답률 :            0.9649122807017544
# RidgeClassifier 의 정답률 :                   0.9824561403508771
# RidgeClassifierCV 의 정답률 :                 0.9824561403508771
# SGDClassifier 의 정답률 :                     0.9649122807017544
# SVC 의 정답률 :                               0.956140350877193
# StackingClassifier 은 없는 놈!
# VotingClassifier 은 없는 놈!
