2021-01-29
range: m14 ~ m23

keywords: 
(1) Pipeline / make_pipeline (scaler + estimator), 
(2) Feature Importances (fit 이후에 가능)

REVIEW
DAY1:
머신러닝 전체에 대해서 리뷰를 좀 해보자.
첫 인상: 겁나 빠르고 성능도 중상타 하네~ (쌔빠지게 튜닝 한 것 보다 낫네)
XOR 문제, DNN으로 해결 (30년 전 장비가 못 따라줘서 미친 짓이었음)
iris로 대표적인 모델들을 봤다. 여러 DecisionTree가 엮인 게 RandomForest임 (ensemble)
분류 -> Classifier (accuracy); 회귀 -> Regressor (r2)
머신러닝은 4단계로 구성 data -> model -> fit -> score and predict (score는 evaluate와 비유됨)

'모든 모델 확인'
all_estimators: 모든 모델을 뽑아서 평가를 해봤다, 요구하는 값과 불일치하는 것을 처리하기 위해서 예외처리 (try, except) 배움

DAY2:
'train_test_split의 진화'
KFold 목적: validation하는데 전체적으로 다 하는 것보다 예를 들어 5등분으로 하여 구간마다 검증
cross_val_score에 모델, 데이터, cv=kfold로 잡아준다; 이름에 score 들어가 있으니까 fit도 되는 거 알겠지
all_estimators에 kfold를 적용했고, 시간이 많이 걸리지 않았다

'파라미터의 자동화'
본격적인 활용도가 높아지는 부분은 GridSearch부터다; 파라미터 list of dict로 만들어주고, kfold도 적용해줬음. 그리고 RandomForest 모델을 썼음.
파라미터에 대해서 일부 설명 들어갔음 (depth, n_jobs), 파라미터를 직렬로 이었음 (곱셈)
GridSearch의 문제점은 너무 많이 돌아가서 시간 소비가 컸음 -> RandomizedSearchCV 등장 (지금은 그나마 빠르지만 나중에 keras에 엮으면 느려짐)
n_iter (default: 10) * cross_val_score = total cycle

=====

(1) Pipeline

'전처리와 모델 이어주기'
from sklearn.pipeline import Pipeline, make_pipeline
scaler와 estimator을 엮어주는 것이 pipeline이다
cf> m14_pipeline_iris.py

이후, Search로 전체 묶어줌, 
ex> GridSearch(pipe(scaler + estimator), parameters, cv)
cf> m15_pipeline_gridSearch.py, m16_pipeline_randomizedSearch_iris.py

여기에서 한 단계 더 발전하면 m17_nested_cv.py인데,
표현은 nested로 되어있지만, 사실상 어제 배운 cross_val_score을 잘 떠올려보면
m15, m16에서 fit하고 score을 해준 거를 cross_val_score로 치환해준 것임. (간단하쥬?)
그런데 cross_val_score = fit + score + cv 인거 기억한다면, cv가 부가적으로 더 들어가서
nested라고 표현할 수 있는 것임.
cf> m17_nested_cv.py, m18_nested_cv_pipeline_wine.py 참고

** 이거 한 줄만 기억하자!!! **
### 최종정리: Search(pipe(scaler + estimator), parameters, cv); cross_val_score(estimator, X_train, y_train, cv) ###


(2) Feature_importances

오후에는 "중요"한 거 한다고 했지, 이제부터는 "중요"한 거 할거니까 정신 바짝 차려라
plot 해보고 feature importances 파악하고 중요하지 않은 feature를 줄여라 -> m20_FI_plot_func_iris.py
각종 Tree 모델에 적용하는 실습 -> m21_FI_TreeModels_cancer.py (DecisionTree, RandomForest, GradientBoosting)
# 원래 21, 22, 23에 각 모델이 있었는데 한 파일로 합침

오늘 FI는 수작업으로 피쳐를 줄이는 거라면, 내일 배울 PCA는 자동 압축
